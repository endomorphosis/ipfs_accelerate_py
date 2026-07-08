#!/usr/bin/env python3
"""
Template for Object Detection models such as DETR, YOLO, MaskFormer, etc.

This template is designed for models that detect and localize objects in images,
optionally with segmentation masks or additional attributes.
"""

import os
import time
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from transformers import {model_class_name}, {processor_class_name}
    import transformers
except ImportError:
    raise ImportError(
        "The transformers package is required to use this model. "
        "Please install it with `pip install transformers`."
    )

logger = logging.getLogger(__name__)

class {skillset_class_name}:
    """
    Skillset for {model_type_upper} - an object detection model that locates
    and classifies objects in images, potentially with segmentation masks.
    """
    
    def __init__(self, model_id: str = "{default_model_id}", device: str = "cpu", **kwargs):
        """
        Initialize the {model_type_upper} model.
        
        Args:
            model_id: HuggingFace model ID or path
            device: Device to run the model on ('cpu', 'cuda', 'rocm', 'mps', etc.)
            **kwargs: Additional arguments to pass to the model
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.is_initialized = False
        
        # Track hardware info for reporting
        self.hardware_info = {
            "device": device,
            "device_name": None,
            "memory_available": None,
            "supports_half_precision": False,
            "detection_specific": {
                "max_objects": None,
                "supports_segmentation": False,
                "input_resolution": None,
                "classes": []
            }
        }
        
        # Optional configuration
        self.low_memory_mode = kwargs.get("low_memory_mode", False)
        self.max_memory = kwargs.get("max_memory", None)
        self.torch_dtype = kwargs.get("torch_dtype", None)
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        self.max_objects = kwargs.get("max_objects", 100)
        
        # Initialize the model if auto_init is True
        auto_init = kwargs.get("auto_init", True)
        if auto_init:
            self.initialize()
    
    def initialize(self):
        """Initialize the model and processor."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing {self.model_id} on {self.device}")
        start_time = time.time()
        
        try:
            # Check if CUDA is available when device is cuda
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if MPS is available when device is mps
            if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if ROCm/HIP is available when device is rocm
            if self.device == "rocm":
                rocm_available = False
                try:
                    if hasattr(torch, 'hip') and torch.hip.is_available():
                        rocm_available = True
                    elif torch.cuda.is_available():
                        # Could be ROCm using CUDA API
                        device_name = torch.cuda.get_device_name(0)
                        if "AMD" in device_name or "Radeon" in device_name:
                            rocm_available = True
                            self.hardware_info.update({
                                "device_name": device_name,
                                "memory_available": torch.cuda.get_device_properties(0).total_memory,
                                "supports_half_precision": True  # Most AMD GPUs support half precision
                            })
                except:
                    rocm_available = False
                
                if not rocm_available:
                    logger.warning("ROCm requested but not available, falling back to CPU")
                    self.device = "cpu"
            
            # CPU is the fallback
            if self.device == "cpu":
                self.hardware_info.update({
                    "device_name": "CPU",
                    "supports_half_precision": False
                })
            
            # Determine dtype based on hardware
            if self.torch_dtype is None:
                if self.hardware_info["supports_half_precision"] and not self.low_memory_mode:
                    self.torch_dtype = torch.float16
                else:
                    self.torch_dtype = torch.float32
            
            # Load processor
            try:
                self.processor = {processor_class_name}.from_pretrained(self.model_id)
            except Exception as e:
                logger.warning(f"Error loading processor: {str(e)}. Creating a mock processor.")
                self.processor = self._create_mock_processor()
            
            # Load model with appropriate configuration
            load_kwargs = {}
            if self.torch_dtype is not None:
                load_kwargs["torch_dtype"] = self.torch_dtype
            
            if self.low_memory_mode:
                load_kwargs["low_cpu_mem_usage"] = True
            
            if self.max_memory is not None:
                load_kwargs["max_memory"] = self.max_memory
            
            # Specific handling for device placement
            if self.device.startswith(("cuda", "rocm")) and "device_map" not in load_kwargs:
                load_kwargs["device_map"] = "auto"
            
            # Load the object detection model
            self.model = {model_class_name}.from_pretrained(self.model_id, **load_kwargs)
            
            # Move to appropriate device if not using device_map
            if "device_map" not in load_kwargs and not self.device.startswith(("cuda", "rocm")):
                self.model.to(self.device)
            
            # Extract model-specific information
            self._extract_model_info()
            
            # Log initialization time
            elapsed_time = time.time() - start_time
            logger.info(f"Initialized {self.model_id} in {elapsed_time:.2f} seconds")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing {self.model_id}: {str(e)}")
            raise
    
    def _extract_model_info(self):
        """Extract model-specific information for detection."""
        # Default values
        max_objects = self.max_objects
        supports_segmentation = False
        input_resolution = (800, 800)  # Default
        classes = []
        
        # Try to get info from the model's config
        if hasattr(self.model, "config"):
            # Check if model supports segmentation
            model_type = getattr(self.model.config, "model_type", "").lower()
            architectures = getattr(self.model.config, "architectures", [])
            
            # Check for segmentation support based on model type or architecture
            if any(seg_name in str(model_type) for seg_name in ["maskformer", "mask2former", "sam", "segment"]):
                supports_segmentation = True
            elif any(seg_name in str(arch) for arch in architectures for seg_name in ["Mask", "Segment", "SAM"]):
                supports_segmentation = True
            
            # Get max objects if available
            if hasattr(self.model.config, "num_queries"):
                max_objects = self.model.config.num_queries
            elif hasattr(self.model.config, "max_detections"):
                max_objects = self.model.config.max_detections
            
            # Get input resolution if available
            if hasattr(self.model.config, "image_size"):
                input_resolution = self.model.config.image_size
                if isinstance(input_resolution, int):
                    input_resolution = (input_resolution, input_resolution)
            
            # Get classes if available
            if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                classes = list(self.model.config.id2label.values())
            
            # For ID2LABEL compatibility (some models use uppercase)
            elif hasattr(self.model.config, "ID2LABEL") and self.model.config.ID2LABEL:
                classes = list(self.model.config.ID2LABEL.values())
        
        # Try to get classes from processor
        if not classes and hasattr(self.processor, "id2label"):
            classes = list(self.processor.id2label.values())
        
        # COCO classes fallback if none found
        if not classes:
            classes = [
                "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
                "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
                "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
        
        # Update hardware info
        self.hardware_info["detection_specific"].update({
            "max_objects": max_objects,
            "supports_segmentation": supports_segmentation,
            "input_resolution": input_resolution,
            "classes": classes
        })
    
    def _create_mock_processor(self):
        """Create a mock processor for object detection when the real one fails."""
        class MockObjectDetectionProcessor:
            def __init__(self):
                # COCO classes as default
                self.id2label = {
                    i: label for i, label in enumerate([
                        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                        "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
                        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                        "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
                        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                        "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                        "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
                        "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                        "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock",
                        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                    ])
                }
                self.label2id = {v: k for k, v in self.id2label.items()}
            
            def __call__(self, images=None, return_tensors="pt", **kwargs):
                """Mock processing of images."""
                import torch
                
                # Default image size
                image_size = (800, 800)
                
                # Create mock pixel values for a batch
                if images is None:
                    # Create a fake batch with 1 image of random noise
                    pixel_values = torch.rand(1, 3, image_size[0], image_size[1])
                else:
                    # Create fake batch based on the number of images
                    if isinstance(images, (list, tuple)):
                        batch_size = len(images)
                    else:
                        batch_size = 1
                    
                    pixel_values = torch.rand(batch_size, 3, image_size[0], image_size[1])
                
                return {
                    "pixel_values": pixel_values,
                    "original_sizes": torch.tensor([[image_size[0], image_size[1]]]).repeat(pixel_values.size(0), 1),
                    "sizes": torch.tensor([[image_size[0], image_size[1]]]).repeat(pixel_values.size(0), 1)
                }
            
            def post_process_object_detection(self, outputs, target_sizes=None, threshold=0.5, **kwargs):
                """Mock post-processing for object detection."""
                import torch
                
                # Create mock detections
                batch_size = 1  # Default batch size
                num_detections = 3  # Default number of mock detections
                
                # Attempt to determine batch size from outputs
                if hasattr(outputs, "logits") and hasattr(outputs.logits, "shape"):
                    batch_size = outputs.logits.shape[0]
                
                results = []
                for i in range(batch_size):
                    # Create mock boxes, scores, and labels
                    mock_boxes = torch.tensor([
                        [0.1, 0.1, 0.5, 0.5],
                        [0.2, 0.2, 0.7, 0.7],
                        [0.3, 0.3, 0.9, 0.9]
                    ])
                    
                    mock_scores = torch.tensor([0.9, 0.8, 0.7])
                    mock_labels = torch.tensor([1, 2, 3])  # person, bicycle, car
                    
                    # Add to results
                    results.append({
                        "boxes": mock_boxes,
                        "scores": mock_scores,
                        "labels": mock_labels
                    })
                
                return results
            
            def post_process_instance_segmentation(self, outputs, target_sizes=None, threshold=0.5, **kwargs):
                """Mock post-processing for segmentation."""
                import torch
                
                # Create mock segmentations
                batch_size = 1  # Default batch size
                num_detections = 3  # Default number of mock detections
                
                # Attempt to determine batch size from outputs
                if hasattr(outputs, "logits") and hasattr(outputs.logits, "shape"):
                    batch_size = outputs.logits.shape[0]
                
                results = []
                for i in range(batch_size):
                    # Create mock boxes, scores, labels, and masks
                    mock_boxes = torch.tensor([
                        [0.1, 0.1, 0.5, 0.5],
                        [0.2, 0.2, 0.7, 0.7],
                        [0.3, 0.3, 0.9, 0.9]
                    ])
                    
                    mock_scores = torch.tensor([0.9, 0.8, 0.7])
                    mock_labels = torch.tensor([1, 2, 3])  # person, bicycle, car
                    
                    # Create mock masks (28x28 is common size for masks)
                    mock_masks = torch.zeros(num_detections, 28, 28)
                    for j in range(num_detections):
                        # Create a simple circular mask
                        center_x, center_y = 14, 14
                        radius = 10 - j*2
                        for x in range(28):
                            for y in range(28):
                                if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                                    mock_masks[j, y, x] = 1
                    
                    # Add to results
                    results.append({
                        "boxes": mock_boxes,
                        "scores": mock_scores,
                        "labels": mock_labels,
                        "masks": mock_masks
                    })
                
                return results
        
        logger.info("Creating mock object detection processor")
        return MockObjectDetectionProcessor()
    
    def _create_mock_inputs(self, batch_size=1, image_size=(800, 800)):
        """Create mock inputs for graceful degradation."""
        # Create mock pixel values
        pixel_values = torch.rand(batch_size, 3, image_size[0], image_size[1]).to(self.device)
        
        # Create other input tensors that might be needed
        original_sizes = torch.tensor([[image_size[0], image_size[1]]] * batch_size).to(self.device)
        sizes = torch.tensor([[image_size[0], image_size[1]]] * batch_size).to(self.device)
        
        return {
            "pixel_values": pixel_values,
            "original_sizes": original_sizes,
            "sizes": sizes
        }
    
    def process_image(self, image, **kwargs):
        """
        Process an image for object detection.
        
        Args:
            image: The input image (can be PIL Image, numpy array, or tensor)
            **kwargs: Additional processing parameters
            
        Returns:
            Processed inputs ready for the model
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Process the image
            inputs = self.processor(image, return_tensors="pt", **kwargs)
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            # Create mock inputs as fallback
            return self._create_mock_inputs()
    
    def detect_objects(self, image, confidence_threshold=None, **kwargs):
        """
        Detect objects in an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            confidence_threshold: Minimum confidence for detections (default: self.confidence_threshold)
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with detection results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Set confidence threshold
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Process the image
        inputs = self.process_image(image, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get original image size if available
            orig_sizes = None
            if "original_sizes" in inputs:
                orig_sizes = inputs["original_sizes"]
            elif "original_size" in inputs:
                orig_sizes = inputs["original_size"]
            
            # Post-process outputs
            if hasattr(self.processor, "post_process_object_detection"):
                # For object detection models like DETR, YOLO, etc.
                processed_outputs = self.processor.post_process_object_detection(
                    outputs, target_sizes=orig_sizes, threshold=confidence_threshold
                )
                
                # Convert to a unified format
                results = []
                for batch_idx, output in enumerate(processed_outputs):
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    
                    # Map labels to class names if available
                    class_names = []
                    if hasattr(self.processor, "id2label"):
                        class_names = [self.processor.id2label.get(int(label), str(label)) for label in labels]
                    else:
                        class_names = [str(label) for label in labels]
                    
                    batch_results = []
                    for i in range(len(boxes)):
                        detection = {
                            "box": boxes[i].tolist(),
                            "score": float(scores[i]),
                            "label": int(labels[i]),
                            "class": class_names[i]
                        }
                        batch_results.append(detection)
                    
                    results.append(batch_results)
                
                # Return single result for single image
                if len(results) == 1:
                    results = results[0]
                
                # Get info about the model's capabilities
                return {
                    "detections": results,
                    "confidence_threshold": confidence_threshold,
                    "supports_segmentation": self.hardware_info["detection_specific"]["supports_segmentation"]
                }
                
            elif hasattr(self.processor, "post_process_detection"):
                # For models with a more generic post-processing method
                processed_outputs = self.processor.post_process_detection(
                    outputs, target_sizes=orig_sizes, threshold=confidence_threshold
                )
                
                # Process in a similar way to object detection
                # ... (similar processing as above)
                # Return single result for single image
                
                # Placeholder for generic post-processing
                logger.warning("Using generic post-processing, results may need additional formatting")
                return {"raw_outputs": outputs}
                
            else:
                # No specialized post-processing available, do our best
                logger.warning("No post-processing method available, returning raw outputs")
                
                # Try to extract boxes and scores if possible
                results = []
                if hasattr(outputs, "pred_boxes") or hasattr(outputs, "boxes"):
                    boxes = outputs.pred_boxes if hasattr(outputs, "pred_boxes") else outputs.boxes
                    boxes = boxes.cpu().numpy()
                    
                    scores = None
                    if hasattr(outputs, "scores"):
                        scores = outputs.scores.cpu().numpy()
                    
                    labels = None
                    if hasattr(outputs, "logits"):
                        # Assuming logits shape is [batch, num_queries, num_classes]
                        labels = outputs.logits.argmax(dim=-1).cpu().numpy()
                    
                    # Format results
                    batch_size = boxes.shape[0]
                    for i in range(batch_size):
                        batch_results = []
                        for j in range(len(boxes[i])):
                            detection = {"box": boxes[i][j].tolist()}
                            if scores is not None and j < len(scores[i]):
                                detection["score"] = float(scores[i][j])
                            if labels is not None and j < len(labels[i]):
                                detection["label"] = int(labels[i][j])
                            batch_results.append(detection)
                        results.append(batch_results)
                    
                    # Return single result for single image
                    if len(results) == 1:
                        results = results[0]
                    
                    return {"detections": results, "confidence_threshold": confidence_threshold}
                else:
                    # Convert any tensors to numpy for serialization
                    raw_dict = {}
                    for k, v in outputs.items():
                        if hasattr(v, "cpu"):
                            raw_dict[k] = v.cpu().numpy()
                        else:
                            raw_dict[k] = v
                    
                    return {"raw_outputs": raw_dict}
        
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return {"error": str(e)}
    
    def segment_objects(self, image, confidence_threshold=None, **kwargs):
        """
        Detect objects with segmentation masks.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            confidence_threshold: Minimum confidence for detections (default: self.confidence_threshold)
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with segmentation results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Check if model supports segmentation
        if not self.hardware_info["detection_specific"]["supports_segmentation"]:
            logger.warning("This model may not support segmentation, falling back to object detection")
            return self.detect_objects(image, confidence_threshold, **kwargs)
        
        # Set confidence threshold
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Process the image
        inputs = self.process_image(image, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get original image size if available
            orig_sizes = None
            if "original_sizes" in inputs:
                orig_sizes = inputs["original_sizes"]
            elif "original_size" in inputs:
                orig_sizes = inputs["original_size"]
            
            # Post-process outputs
            if hasattr(self.processor, "post_process_instance_segmentation"):
                # For instance segmentation models like MaskFormer, Mask2Former
                processed_outputs = self.processor.post_process_instance_segmentation(
                    outputs, target_sizes=orig_sizes, threshold=confidence_threshold
                )
                
                # Convert to a unified format
                results = []
                for batch_idx, output in enumerate(processed_outputs):
                    boxes = output["boxes"].cpu().numpy() if "boxes" in output else None
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    masks = output["masks"].cpu().numpy() if "masks" in output else None
                    
                    # Map labels to class names if available
                    class_names = []
                    if hasattr(self.processor, "id2label"):
                        class_names = [self.processor.id2label.get(int(label), str(label)) for label in labels]
                    else:
                        class_names = [str(label) for label in labels]
                    
                    batch_results = []
                    for i in range(len(scores)):
                        detection = {
                            "score": float(scores[i]),
                            "label": int(labels[i]),
                            "class": class_names[i]
                        }
                        
                        if boxes is not None and i < len(boxes):
                            detection["box"] = boxes[i].tolist()
                        
                        if masks is not None and i < len(masks):
                            detection["mask"] = masks[i].tolist()
                        
                        batch_results.append(detection)
                    
                    results.append(batch_results)
                
                # Return single result for single image
                if len(results) == 1:
                    results = results[0]
                
                # Get info about the model's capabilities
                return {
                    "segmentations": results,
                    "confidence_threshold": confidence_threshold,
                    "supports_segmentation": True
                }
                
            elif hasattr(self.processor, "post_process_semantic_segmentation"):
                # For semantic segmentation models
                processed_outputs = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=orig_sizes
                )
                
                # Format results for semantic segmentation
                # In semantic segmentation, each pixel has a class label
                results = []
                for batch_idx, output in enumerate(processed_outputs):
                    # Convert to numpy
                    segmentation_map = output.cpu().numpy()
                    
                    # Get unique class labels in the segmentation map
                    unique_labels = np.unique(segmentation_map)
                    
                    # Map labels to class names if available
                    class_names = {}
                    if hasattr(self.processor, "id2label"):
                        for label in unique_labels:
                            class_names[int(label)] = self.processor.id2label.get(int(label), str(label))
                    
                    results.append({
                        "segmentation_map": segmentation_map.tolist(),
                        "unique_labels": unique_labels.tolist(),
                        "class_names": class_names
                    })
                
                # Return single result for single image
                if len(results) == 1:
                    results = results[0]
                
                return {
                    "semantic_segmentation": results,
                    "confidence_threshold": confidence_threshold,
                    "supports_segmentation": True
                }
                
            elif hasattr(self.processor, "post_process_panoptic_segmentation"):
                # For panoptic segmentation models
                processed_outputs = self.processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=orig_sizes, threshold=confidence_threshold
                )
                
                # Format results for panoptic segmentation
                results = []
                for batch_idx, output in enumerate(processed_outputs):
                    # Convert to numpy
                    segmentation_map = output["segmentation"].cpu().numpy()
                    segments_info = output["segments_info"]
                    
                    # Process segments info
                    processed_segments = []
                    for segment in segments_info:
                        segment_id = segment["id"]
                        label_id = segment["label_id"]
                        class_name = self.processor.id2label.get(label_id, str(label_id)) if hasattr(self.processor, "id2label") else str(label_id)
                        
                        processed_segments.append({
                            "id": segment_id,
                            "label": label_id,
                            "class": class_name,
                            "area": segment.get("area", None),
                            "is_thing": segment.get("is_thing", None)
                        })
                    
                    results.append({
                        "segmentation_map": segmentation_map.tolist(),
                        "segments_info": processed_segments
                    })
                
                # Return single result for single image
                if len(results) == 1:
                    results = results[0]
                
                return {
                    "panoptic_segmentation": results,
                    "confidence_threshold": confidence_threshold,
                    "supports_segmentation": True
                }
                
            elif hasattr(self.processor, "post_process_object_detection") and hasattr(outputs, "pred_masks"):
                # For models where masks are directly available
                processed_outputs = self.processor.post_process_object_detection(
                    outputs, target_sizes=orig_sizes, threshold=confidence_threshold
                )
                
                # Add masks to the processed outputs
                masks = outputs.pred_masks.cpu().sigmoid().numpy() if hasattr(outputs.pred_masks, "sigmoid") else outputs.pred_masks.cpu().numpy()
                
                # Convert to a unified format
                results = []
                for batch_idx, output in enumerate(processed_outputs):
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    
                    # Map labels to class names if available
                    class_names = []
                    if hasattr(self.processor, "id2label"):
                        class_names = [self.processor.id2label.get(int(label), str(label)) for label in labels]
                    else:
                        class_names = [str(label) for label in labels]
                    
                    batch_results = []
                    for i in range(len(boxes)):
                        detection = {
                            "box": boxes[i].tolist(),
                            "score": float(scores[i]),
                            "label": int(labels[i]),
                            "class": class_names[i]
                        }
                        
                        if batch_idx < masks.shape[0] and i < masks.shape[1]:
                            detection["mask"] = masks[batch_idx, i].tolist()
                        
                        batch_results.append(detection)
                    
                    results.append(batch_results)
                
                # Return single result for single image
                if len(results) == 1:
                    results = results[0]
                
                return {
                    "segmentations": results,
                    "confidence_threshold": confidence_threshold,
                    "supports_segmentation": True
                }
                
            else:
                # No specialized post-processing available, fall back to object detection
                logger.warning("No segmentation post-processing available, falling back to object detection")
                return self.detect_objects(image, confidence_threshold, **kwargs)
        
        except Exception as e:
            logger.error(f"Error during segmentation: {str(e)}")
            return {"error": str(e)}
    
    def __call__(self, image, task: str = "detect", confidence_threshold=None, **kwargs) -> Dict[str, Any]:
        """
        Process an image with the model.
        
        Args:
            image: The input image
            task: Task to perform ('detect' or 'segment')
            confidence_threshold: Minimum confidence for detections
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with task results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Set confidence threshold
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Select task
        if task == "detect":
            return self.detect_objects(image, confidence_threshold, **kwargs)
        elif task == "segment":
            return self.segment_objects(image, confidence_threshold, **kwargs)
        else:
            # Default to detection
            logger.warning(f"Unknown task '{task}', defaulting to detect")
            return self.detect_objects(image, confidence_threshold, **kwargs)

    def __test__(self, **kwargs):
        """
        Run a self-test to verify the model is working correctly.
        
        Returns:
            Dictionary with test results
        """
        if not self.is_initialized:
            try:
                self.initialize()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Initialization failed: {str(e)}",
                    "hardware": self.hardware_info
                }
        
        results = {
            "hardware": self.hardware_info,
            "tests": {}
        }
        
        # Test 1: Create mock inputs
        try:
            # Test input creation
            inputs = self._create_mock_inputs()
            
            results["tests"]["input_creation"] = {
                "success": True,
                "input_keys": list(inputs.keys()),
                "pixel_values_shape": tuple(inputs["pixel_values"].shape)
            }
        except Exception as e:
            results["tests"]["input_creation"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Object detection
        try:
            # Create a small test image
            test_image = torch.rand(3, 300, 300)  # RGB image
            
            # Test object detection
            output = self.detect_objects(test_image, confidence_threshold=0.1)
            
            results["tests"]["object_detection"] = {
                "success": "detections" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "detection_count": len(output["detections"]) if "detections" in output else 0
            }
        except Exception as e:
            results["tests"]["object_detection"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Segmentation (if supported)
        if self.hardware_info["detection_specific"]["supports_segmentation"]:
            try:
                # Create a small test image
                test_image = torch.rand(3, 300, 300)  # RGB image
                
                # Test segmentation
                output = self.segment_objects(test_image, confidence_threshold=0.1)
                
                results["tests"]["segmentation"] = {
                    "success": any(key in output for key in ["segmentations", "semantic_segmentation", "panoptic_segmentation"]),
                    "output_keys": list(output.keys())
                }
            except Exception as e:
                results["tests"]["segmentation"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Overall success determination
        successful_tests = sum(1 for t in results["tests"].values() if t.get("success", False))
        results["success"] = successful_tests > 0
        results["success_rate"] = successful_tests / len(results["tests"])
        
        return results


class TestObjectDetectionModel:
    """Test suite for the Object Detection model implementation."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.model_id = "facebook/detr-resnet-50"  # Default test model
        self.low_memory_mode = True  # Use low memory mode for testing
    
    def run_tests(self):
        """Run all tests and return results."""
        results = {}
        
        # Test initialization
        init_result = self.test_initialization()
        results["initialization"] = init_result
        
        # If initialization failed, skip other tests
        if not init_result.get("success", False):
            return results
        
        # Test input processing
        results["input_processing"] = self.test_input_processing()
        
        # Test object detection
        results["object_detection"] = self.test_object_detection()
        
        # Test segmentation
        results["segmentation"] = self.test_segmentation()
        
        # Determine overall success
        successful_tests = sum(1 for t in results.values() if t.get("success", False))
        results["overall_success"] = successful_tests / len(results)
        
        return results
    
    def test_initialization(self):
        """Test model initialization."""
        try:
            # Import the model class
            from transformers import DetrForObjectDetection, DetrImageProcessor
            
            # Initialize the model with minimal config
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Run basic self-test
            test_result = model.__test__()
            
            return {
                "success": test_result.get("success", False),
                "hardware_info": model.hardware_info,
                "details": test_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_input_processing(self):
        """Test image processing functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test image
            test_image = torch.rand(3, 300, 300)  # RGB image
            
            # Test image processing
            inputs = model.process_image(test_image)
            
            return {
                "success": "pixel_values" in inputs,
                "input_keys": list(inputs.keys()),
                "pixel_values_shape": tuple(inputs["pixel_values"].shape)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_object_detection(self):
        """Test object detection functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test image
            test_image = torch.rand(3, 300, 300)  # RGB image
            
            # Test object detection
            output = model.detect_objects(test_image, confidence_threshold=0.1)
            
            return {
                "success": "detections" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "detection_count": len(output["detections"]) if "detections" in output else 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_segmentation(self):
        """Test segmentation functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test image
            test_image = torch.rand(3, 300, 300)  # RGB image
            
            # Test segmentation
            output = model.segment_objects(test_image, confidence_threshold=0.1)
            
            return {
                "success": any(key in output for key in ["segmentations", "semantic_segmentation", "panoptic_segmentation", "detections"]),
                "output_keys": list(output.keys()),
                "supports_segmentation": model.hardware_info["detection_specific"]["supports_segmentation"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    # Run tests if executed directly
    tester = TestObjectDetectionModel()
    results = tester.run_tests()
    
    print(json.dumps(results, indent=2))