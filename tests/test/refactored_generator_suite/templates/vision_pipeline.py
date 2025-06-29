#!/usr/bin/env python3
"""
Vision pipeline template for IPFS Accelerate Python.

This module implements the pipeline template for vision-based models.
"""

from typing import Dict, Any, Callable, Tuple, Optional, List, Union
from templates.base_pipeline import BasePipelineTemplate


class VisionPipelineTemplate(BasePipelineTemplate):
    """Vision pipeline template implementation."""
    
    def __init__(self):
        """Initialize the vision pipeline template."""
        super().__init__()
        self.pipeline_type = "vision"
        self.input_type = "image"
        self.output_type = "classification"
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 16
    
    def get_import_statements(self) -> str:
        """Get vision-specific import statements."""
        return """
# Vision-specific imports
import os
import json
import numpy as np
import PIL
from PIL import Image
import torch
from typing import List, Dict, Union, Any
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get vision-specific preprocessing code."""
        if task_type == "image_classification":
            return """
# Check if input is already a list of images
if not isinstance(images, list):
    images = [images]

# Process each image
processed_images = []
for img in images:
    # Convert to PIL if needed
    if isinstance(img, str):
        # Load from path
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        # Convert from numpy array
        img = Image.fromarray(img)
    elif not isinstance(img, PIL.Image.Image):
        raise ValueError(f"Unsupported image type: {type(img)}")
    
    # Apply preprocessing
    processed_img = processor(images=img, return_tensors="pt")
    processed_images.append(processed_img)

# Combine batch
if len(processed_images) == 1:
    inputs = processed_images[0]
else:
    # Batch processing for multiple images
    keys = processed_images[0].keys()
    inputs = {k: torch.cat([img[k] for img in processed_images]) for k in keys}

# Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "image_segmentation":
            return """
# Check if input is already a list of images
if not isinstance(images, list):
    images = [images]

# Process each image
processed_images = []
for img in images:
    # Convert to PIL if needed
    if isinstance(img, str):
        # Load from path
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        # Convert from numpy array
        img = Image.fromarray(img)
    elif not isinstance(img, PIL.Image.Image):
        raise ValueError(f"Unsupported image type: {type(img)}")
    
    # Apply preprocessing for segmentation
    processed_img = processor(images=img, return_tensors="pt")
    processed_images.append(processed_img)

# Combine batch
if len(processed_images) == 1:
    inputs = processed_images[0]
else:
    # Batch processing for multiple images
    keys = processed_images[0].keys()
    inputs = {k: torch.cat([img[k] for img in processed_images]) for k in keys}

# Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "object_detection":
            return """
# Check if input is already a list of images
if not isinstance(images, list):
    images = [images]

# Process each image
processed_images = []
original_sizes = []
for img in images:
    # Convert to PIL if needed
    if isinstance(img, str):
        # Load from path
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        # Convert from numpy array
        img = Image.fromarray(img)
    elif not isinstance(img, PIL.Image.Image):
        raise ValueError(f"Unsupported image type: {type(img)}")
    
    # Save original size for bounding box scaling later
    original_sizes.append((img.width, img.height))
    
    # Apply preprocessing for object detection
    processed_img = processor(images=img, return_tensors="pt")
    processed_images.append(processed_img)

# Combine batch
if len(processed_images) == 1:
    inputs = processed_images[0]
else:
    # Batch processing for multiple images
    keys = processed_images[0].keys()
    inputs = {k: torch.cat([img[k] for img in processed_images]) for k in keys}

# Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        else:
            return f"""
# Default preprocessing for {task_type}
# Check if input is already a list of images
if not isinstance(images, list):
    images = [images]

# Process each image
processed_images = []
for img in images:
    # Convert to PIL if needed
    if isinstance(img, str):
        # Load from path
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        # Convert from numpy array
        img = Image.fromarray(img)
    elif not isinstance(img, PIL.Image.Image):
        raise ValueError(f"Unsupported image type: {{type(img)}}")
    
    # Apply preprocessing
    processed_img = processor(images=img, return_tensors="pt")
    processed_images.append(processed_img)

# Combine batch
if len(processed_images) == 1:
    inputs = processed_images[0]
else:
    # Batch processing for multiple images
    keys = processed_images[0].keys()
    inputs = {{k: torch.cat([img[k] for img in processed_images]) for k in keys}}

# Move to device
inputs = {{k: v.to(device) for k, v in inputs.items()}}
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get vision-specific postprocessing code."""
        if task_type == "image_classification":
            return """
# Postprocess classification results
logits = outputs.logits
probs = torch.nn.functional.softmax(logits, dim=-1)
probs = probs.cpu().numpy()

# Get top predictions
top_k = min(5, probs.shape[-1])
top_probs, top_indices = [], []
for i in range(probs.shape[0]):
    # Get top-k probabilities and indices
    instance_probs = probs[i]
    indices = np.argsort(instance_probs)[-top_k:][::-1]
    values = instance_probs[indices]
    top_indices.append(indices.tolist())
    top_probs.append(values.tolist())

# Map indices to class labels if available
class_labels = []
if hasattr(processor, 'id2label'):
    for indices in top_indices:
        labels = [processor.id2label[idx] if idx in processor.id2label else f"LABEL_{idx}" for idx in indices]
        class_labels.append(labels)
else:
    class_labels = [[f"CLASS_{idx}" for idx in indices] for indices in top_indices]
"""
        elif task_type == "image_segmentation":
            return """
# Postprocess segmentation results
logits = outputs.logits  # Shape: [batch_size, num_classes, height, width]
pred_masks = logits.argmax(dim=1)  # Shape: [batch_size, height, width]
pred_masks = pred_masks.cpu().numpy()

# Convert to RGB segmentation maps if available
if hasattr(processor, 'id2label'):
    segmentation_maps = []
    for mask in pred_masks:
        # Create an RGB segmentation map
        segmentation_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label_id, label in processor.id2label.items():
            # Assign a color to each class
            # Simple hash function to get consistent colors
            color = [(hash(label) % 255), (hash(label + 'a') % 255), (hash(label + 'b') % 255)]
            segmentation_map[mask == label_id] = color
        segmentation_maps.append(segmentation_map)
else:
    # If there are no labels, just normalize the masks for visualization
    segmentation_maps = [(mask * 255 / mask.max()).astype(np.uint8) for mask in pred_masks]
"""
        elif task_type == "object_detection":
            return """
# Postprocess object detection results
logits = outputs.logits  # Shape: [batch_size, num_boxes, num_classes]
boxes = outputs.pred_boxes  # Shape: [batch_size, num_boxes, 4]

# Convert logits to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Get the most likely class and its probability for each box
max_probs, labels = probs.max(dim=-1)

# Move tensors to CPU and convert to numpy
boxes = boxes.cpu().numpy()
labels = labels.cpu().numpy()
scores = max_probs.cpu().numpy()

# Apply score threshold
threshold = 0.5
detections = []
for i in range(len(boxes)):
    # Get batch item detections
    batch_boxes = boxes[i]
    batch_labels = labels[i]
    batch_scores = scores[i]
    
    # Filter by threshold
    mask = batch_scores > threshold
    filtered_boxes = batch_boxes[mask]
    filtered_labels = batch_labels[mask]
    filtered_scores = batch_scores[mask]
    
    # Scale boxes to original image size
    orig_width, orig_height = original_sizes[i]
    scaled_boxes = []
    for box in filtered_boxes:
        # Box format: [x1, y1, x2, y2] (normalized)
        x1, y1, x2, y2 = box
        scaled_box = [
            int(x1 * orig_width),
            int(y1 * orig_height),
            int(x2 * orig_width),
            int(y2 * orig_height)
        ]
        scaled_boxes.append(scaled_box)
    
    # Map class indices to labels if available
    if hasattr(processor, 'id2label'):
        filtered_labels = [processor.id2label.get(int(l), f"LABEL_{l}") for l in filtered_labels]
    
    # Combine results
    batch_detections = []
    for box, label, score in zip(scaled_boxes, filtered_labels, filtered_scores):
        batch_detections.append({
            "box": box,
            "label": label,
            "score": float(score)
        })
    
    detections.append(batch_detections)
"""
        else:
            return f"""
# Default postprocessing for {task_type}
results = outputs
result_data = {{}}

# Try to extract common types of outputs
if hasattr(outputs, 'logits'):
    result_data['logits'] = outputs.logits.cpu().numpy().tolist()

if hasattr(outputs, 'last_hidden_state'):
    result_data['features'] = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

if hasattr(outputs, 'pred_boxes'):
    result_data['boxes'] = outputs.pred_boxes.cpu().numpy().tolist()

if not result_data:
    # Fallback: just convert everything we can
    for key, value in outputs.__dict__.items():
        if hasattr(value, 'cpu'):
            result_data[key] = value.cpu().numpy().tolist()
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get vision-specific result formatting code."""
        if task_type == "image_classification":
            return """
# Format classification results
result = {"success": True, "device": device, "hardware": hardware_label}

# Add classification results
classifications = []
for i in range(len(top_indices)):
    instance_result = {
        "top_classes": class_labels[i],
        "probabilities": top_probs[i],
        "indices": top_indices[i]
    }
    classifications.append(instance_result)

# If single image, return first result
if len(classifications) == 1:
    result["classification"] = classifications[0]
else:
    result["batch_classifications"] = classifications

return result
"""
        elif task_type == "image_segmentation":
            return """
# Format segmentation results
result = {"success": True, "device": device, "hardware": hardware_label}

# Add segmentation results
if len(segmentation_maps) == 1:
    result["segmentation_map"] = segmentation_maps[0].tolist() if isinstance(segmentation_maps[0], np.ndarray) else segmentation_maps[0]
    result["pred_mask"] = pred_masks[0].tolist()
else:
    result["batch_segmentation_maps"] = [m.tolist() if isinstance(m, np.ndarray) else m for m in segmentation_maps]
    result["batch_pred_masks"] = [m.tolist() for m in pred_masks]

# Include class information if available
if hasattr(processor, 'id2label'):
    result["classes"] = processor.id2label

return result
"""
        elif task_type == "object_detection":
            return """
# Format object detection results
result = {"success": True, "device": device, "hardware": hardware_label}

# Add detection results
if len(detections) == 1:
    result["detections"] = detections[0]
else:
    result["batch_detections"] = detections

return result
"""
        else:
            return f"""
# Default formatting for {task_type}
result = {{"success": True, "device": device, "hardware": hardware_label}}

# Add whatever results we have
for key, value in result_data.items():
    result[key] = value

return result
"""
    
    def get_mock_input_code(self) -> str:
        """Get vision-specific mock input code."""
        return """
# Mock image input (create a simple gradient image)
import numpy as np
from PIL import Image

def create_mock_image(width=224, height=224):
    # Create a simple gradient test image
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    image = np.stack([xv, yv, np.zeros_like(xv)], axis=-1)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

mock_input = create_mock_image()
"""
    
    def get_mock_output_code(self) -> str:
        """Get vision-specific mock output code."""
        return """
# Mock vision output
mock_classification = {
    "top_classes": ["cat", "dog", "bird", "fish", "horse"],
    "probabilities": [0.7, 0.2, 0.05, 0.03, 0.02],
    "indices": [0, 1, 2, 3, 4]
}

mock_segmentation = np.random.randint(0, 10, (224, 224), dtype=np.uint8)

mock_detections = [
    {"box": [10, 10, 100, 100], "label": "cat", "score": 0.9},
    {"box": [150, 150, 200, 200], "label": "dog", "score": 0.8}
]

mock_output = {"classification": mock_classification}
"""

    def get_pipeline_utilities(self) -> str:
        """Get vision-specific utility functions."""
        return """
# Vision pipeline utilities
def resize_image(image, target_size=(224, 224)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.resize(target_size)

def convert_segmentation_to_image(segmentation_map):
    # Convert a segmentation map to a colorful image
    if isinstance(segmentation_map, list):
        segmentation_map = np.array(segmentation_map)
    
    # Normalize to 0-255 range if not already
    if segmentation_map.max() > 1:
        segmentation_map = segmentation_map.astype(np.float32) / segmentation_map.max() * 255
    
    if len(segmentation_map.shape) == 2:
        # Single channel - convert to colormap
        from matplotlib import cm
        colored_map = cm.viridis(segmentation_map.astype(np.float32) / 255.0)
        colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(colored_map)
    
    return Image.fromarray(segmentation_map.astype(np.uint8))

def draw_detection_boxes(image, detections):
    # Draw bounding boxes on an image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    draw_image = image.copy()
    draw = PIL.ImageDraw.Draw(draw_image)
    
    for det in detections:
        box = det["box"]
        label = det["label"]
        score = det["score"]
        
        # Draw box
        draw.rectangle(box, outline="red", width=2)
        
        # Draw label
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1] - 10), label_text, fill="red")
    
    return draw_image
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check vision pipeline compatibility with architecture type."""
        # Vision pipeline is compatible with vision-based architectures
        return arch_type in [
            "vision",
            "vision-encoder"
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check vision pipeline compatibility with task type."""
        # Vision pipeline is compatible with vision-based tasks
        return task_type in [
            "image_classification",
            "image_segmentation",
            "object_detection",
            "instance_segmentation",
            "semantic_segmentation",
            "panoptic_segmentation"
        ]