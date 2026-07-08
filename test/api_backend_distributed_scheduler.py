#!/usr/bin/env python
"""
API Backend Distributed Scheduler

This module provides specialized scheduling for API backends in the distributed testing framework,
with capabilities for:

1. API-specific resource allocation based on rate limits and token usage
2. Intelligent task distribution to optimize API backend utilization
3. Dynamic adaptation to API performance and availability
4. Cost-aware scheduling to minimize API usage costs

This scheduler extends the standard AdvancedScheduler with API-specific capabilities.
"""

import os
import sys
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
import uuid
import heapq
import random
import math
from datetime import datetime, timedelta

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import distributed testing framework components
from test.distributed_testing.advanced_scheduling import AdvancedScheduler, Task, Worker
from test.distributed_testing.load_balancer import LoadBalancer
from test.distributed_testing.circuit_breaker import CircuitBreaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_scheduler")


class APIRateLimitStrategy(Enum):
    """Strategy for handling API rate limits."""
    STRICT = auto()          # Never exceed rate limits
    ADAPTIVE = auto()        # Adaptively approach limits based on success
    BURST_THROTTLE = auto()  # Allow bursts but throttle overall


class APICostStrategy(Enum):
    """Strategy for optimizing API costs."""
    MINIMIZE_COST = auto()       # Prioritize lowest cost APIs
    BALANCED = auto()            # Balance cost with performance
    PERFORMANCE_FIRST = auto()   # Prioritize performance over cost


@dataclass
class APIRateLimit:
    """Class to track and enforce API rate limits."""
    requests_per_minute: int = 0
    requests_per_hour: int = 0
    tokens_per_minute: int = 0
    max_tokens_per_request: int = 0
    concurrent_requests: int = 0
    
    # Runtime tracking
    minute_request_times: List[float] = field(default_factory=list)
    hour_request_times: List[float] = field(default_factory=list)
    minute_token_usage: int = 0
    last_minute_reset: float = field(default_factory=time.time)
    last_hour_reset: float = field(default_factory=time.time)
    current_concurrent: int = 0
    
    def can_make_request(self, tokens: int = 0) -> bool:
        """Check if a request can be made within rate limits."""
        now = time.time()
        
        # Update tracking data
        self._update_tracking(now)
        
        # Check concurrent request limit
        if self.concurrent_requests > 0 and self.current_concurrent >= self.concurrent_requests:
            return False
        
        # Check requests per minute
        if self.requests_per_minute > 0 and len(self.minute_request_times) >= self.requests_per_minute:
            return False
        
        # Check requests per hour
        if self.requests_per_hour > 0 and len(self.hour_request_times) >= self.requests_per_hour:
            return False
        
        # Check tokens per minute
        if self.tokens_per_minute > 0 and tokens > 0:
            if self.minute_token_usage + tokens > self.tokens_per_minute:
                return False
        
        # Check max tokens per request
        if self.max_tokens_per_request > 0 and tokens > self.max_tokens_per_request:
            return False
        
        return True
    
    def record_request(self, tokens: int = 0) -> None:
        """Record that a request was made."""
        now = time.time()
        
        # Update tracking
        self._update_tracking(now)
        
        # Record request
        self.minute_request_times.append(now)
        self.hour_request_times.append(now)
        self.minute_token_usage += tokens
        self.current_concurrent += 1
    
    def record_request_completed(self) -> None:
        """Record that a request has completed."""
        self.current_concurrent = max(0, self.current_concurrent - 1)
    
    def _update_tracking(self, now: float) -> None:
        """Update tracking data based on current time."""
        # Check if we need to reset minute counters
        if now - self.last_minute_reset >= 60:
            # Reset minute counters
            self.minute_request_times = [t for t in self.minute_request_times if now - t < 60]
            self.minute_token_usage = 0
            self.last_minute_reset = now
        
        # Check if we need to reset hour counters
        if now - self.last_hour_reset >= 3600:
            # Reset hour counters
            self.hour_request_times = [t for t in self.hour_request_times if now - t < 3600]
            self.last_hour_reset = now
    
    def get_utilization(self) -> Dict[str, float]:
        """Get current utilization as a percentage of limits."""
        now = time.time()
        self._update_tracking(now)
        
        utilization = {}
        
        if self.requests_per_minute > 0:
            utilization["minute_requests"] = len(self.minute_request_times) / self.requests_per_minute
        
        if self.requests_per_hour > 0:
            utilization["hour_requests"] = len(self.hour_request_times) / self.requests_per_hour
        
        if self.tokens_per_minute > 0:
            utilization["minute_tokens"] = self.minute_token_usage / self.tokens_per_minute
        
        if self.concurrent_requests > 0:
            utilization["concurrent"] = self.current_concurrent / self.concurrent_requests
        
        return utilization


@dataclass
class APICostProfile:
    """Class to track API usage costs."""
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    cost_per_request: float = 0.0
    
    # Runtime tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    estimated_cost: float = 0.0
    
    def record_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record API usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        # Update estimated cost
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output_tokens
        request_cost = self.cost_per_request
        
        self.estimated_cost += input_cost + output_cost + request_cost
    
    def estimate_cost(self, input_tokens: int = 0, output_tokens: int = 0) -> float:
        """Estimate cost for a specific usage."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output_tokens
        request_cost = self.cost_per_request
        
        return input_cost + output_cost + request_cost
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get a cost report."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "estimated_cost": self.estimated_cost,
            "cost_per_1k_input_tokens": self.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": self.cost_per_1k_output_tokens,
            "cost_per_request": self.cost_per_request
        }


@dataclass
class APIPerformanceProfile:
    """Class to track API performance metrics."""
    recent_latencies: List[float] = field(default_factory=list)
    recent_success_rates: List[float] = field(default_factory=list)
    recent_throughputs: List[float] = field(default_factory=list)
    historical_performance: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    max_history_length: int = 100
    
    def record_latency(self, latency: float) -> None:
        """Record a latency measurement."""
        self.recent_latencies.append(latency)
        
        # Keep history bounded
        if len(self.recent_latencies) > self.max_history_length:
            self.recent_latencies = self.recent_latencies[-self.max_history_length:]
        
        # Add to historical data
        if "latency" not in self.historical_performance:
            self.historical_performance["latency"] = []
        
        self.historical_performance["latency"].append((time.time(), latency))
        
        # Keep historical data bounded
        if len(self.historical_performance["latency"]) > self.max_history_length:
            self.historical_performance["latency"] = self.historical_performance["latency"][-self.max_history_length:]
    
    def record_success_rate(self, success_rate: float) -> None:
        """Record a success rate measurement."""
        self.recent_success_rates.append(success_rate)
        
        # Keep history bounded
        if len(self.recent_success_rates) > self.max_history_length:
            self.recent_success_rates = self.recent_success_rates[-self.max_history_length:]
        
        # Add to historical data
        if "success_rate" not in self.historical_performance:
            self.historical_performance["success_rate"] = []
        
        self.historical_performance["success_rate"].append((time.time(), success_rate))
        
        # Keep historical data bounded
        if len(self.historical_performance["success_rate"]) > self.max_history_length:
            self.historical_performance["success_rate"] = self.historical_performance["success_rate"][-self.max_history_length:]
    
    def record_throughput(self, throughput: float) -> None:
        """Record a throughput measurement."""
        self.recent_throughputs.append(throughput)
        
        # Keep history bounded
        if len(self.recent_throughputs) > self.max_history_length:
            self.recent_throughputs = self.recent_throughputs[-self.max_history_length:]
        
        # Add to historical data
        if "throughput" not in self.historical_performance:
            self.historical_performance["throughput"] = []
        
        self.historical_performance["throughput"].append((time.time(), throughput))
        
        # Keep historical data bounded
        if len(self.historical_performance["throughput"]) > self.max_history_length:
            self.historical_performance["throughput"] = self.historical_performance["throughput"][-self.max_history_length:]
    
    def get_avg_latency(self, lookback: int = None) -> Optional[float]:
        """Get average latency, optionally with a lookback window."""
        if not self.recent_latencies:
            return None
        
        if lookback and len(self.recent_latencies) > lookback:
            values = self.recent_latencies[-lookback:]
        else:
            values = self.recent_latencies
        
        return sum(values) / len(values)
    
    def get_avg_success_rate(self, lookback: int = None) -> Optional[float]:
        """Get average success rate, optionally with a lookback window."""
        if not self.recent_success_rates:
            return None
        
        if lookback and len(self.recent_success_rates) > lookback:
            values = self.recent_success_rates[-lookback:]
        else:
            values = self.recent_success_rates
        
        return sum(values) / len(values)
    
    def get_avg_throughput(self, lookback: int = None) -> Optional[float]:
        """Get average throughput, optionally with a lookback window."""
        if not self.recent_throughputs:
            return None
        
        if lookback and len(self.recent_throughputs) > lookback:
            values = self.recent_throughputs[-lookback:]
        else:
            values = self.recent_throughputs
        
        return sum(values) / len(values)
    
    def get_performance_score(self) -> float:
        """Calculate a performance score based on recent metrics."""
        score = 0.0
        components = 0
        
        # Add latency component (lower is better)
        avg_latency = self.get_avg_latency()
        if avg_latency is not None:
            # Convert to a score where lower latency is better
            # Scale between 0-1 where 1 is best (low latency)
            # Assume 10s is the upper bound for a terrible score
            latency_score = max(0, 1 - (avg_latency / 10))
            score += latency_score
            components += 1
        
        # Add success rate component (higher is better)
        avg_success = self.get_avg_success_rate()
        if avg_success is not None:
            score += avg_success  # Already 0-1
            components += 1
        
        # Add throughput component (higher is better)
        avg_throughput = self.get_avg_throughput()
        if avg_throughput is not None:
            # Scale throughput to 0-1 where 1 is best (high throughput)
            # Assume 10 req/s is an excellent throughput (score of 1)
            throughput_score = min(1, avg_throughput / 10)
            score += throughput_score
            components += 1
        
        # Return average score
        return score / components if components > 0 else 0.5  # Default to neutral if no data


@dataclass
class APIBackendProfile:
    """Complete profile for an API backend."""
    api_type: str
    rate_limits: APIRateLimit = field(default_factory=APIRateLimit)
    cost_profile: APICostProfile = field(default_factory=APICostProfile)
    performance: APIPerformanceProfile = field(default_factory=APIPerformanceProfile)
    circuit_breaker: Optional[CircuitBreaker] = None
    last_update: float = field(default_factory=time.time)
    last_success: Optional[float] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    models: Set[str] = field(default_factory=set)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize circuit breaker if not provided."""
        if self.circuit_breaker is None:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                reset_timeout=300,  # 5 minutes
                half_open_timeout=60  # 1 minute
            )
    
    def can_handle_request(self, tokens: int = 0) -> Tuple[bool, str]:
        """Check if this backend can handle a request."""
        # Check circuit breaker first
        if not self.circuit_breaker.allow_request():
            return False, "circuit_breaker_open"
        
        # Check rate limits
        if not self.rate_limits.can_make_request(tokens):
            return False, "rate_limit_exceeded"
        
        return True, "ok"
    
    def record_request_start(self, tokens: int = 0) -> None:
        """Record that a request is starting."""
        self.rate_limits.record_request(tokens)
        self.total_requests += 1
    
    def record_request_success(self, latency: float, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record a successful request."""
        self.rate_limits.record_request_completed()
        self.successful_requests += 1
        self.last_success = time.time()
        
        # Update performance profile
        self.performance.record_latency(latency)
        self.performance.record_success_rate(1.0)  # This single request was successful
        
        # Update cost profile
        self.cost_profile.record_usage(input_tokens, output_tokens)
        
        # Update circuit breaker
        self.circuit_breaker.record_success()
    
    def record_request_failure(self, error: str) -> None:
        """Record a failed request."""
        self.rate_limits.record_request_completed()
        self.failed_requests += 1
        
        # Update performance profile
        self.performance.record_success_rate(0.0)  # This single request failed
        
        # Update circuit breaker
        self.circuit_breaker.record_failure()
    
    def get_success_rate(self) -> float:
        """Get the overall success rate."""
        if self.total_requests == 0:
            return 1.0  # Default to perfect if no data
        
        return self.successful_requests / self.total_requests
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of this API backend."""
        now = time.time()
        
        return {
            "api_type": self.api_type,
            "circuit_breaker_status": self.circuit_breaker.state.name,
            "success_rate": self.get_success_rate(),
            "total_requests": self.total_requests,
            "rate_limit_utilization": self.rate_limits.get_utilization(),
            "last_success_age": now - self.last_success if self.last_success else None,
            "performance_score": self.performance.get_performance_score(),
            "avg_latency": self.performance.get_avg_latency(),
            "cost_to_date": self.cost_profile.estimated_cost,
            "models": list(self.models),
            "capabilities": self.capabilities
        }
    
    def get_cost_effectiveness(self) -> float:
        """Calculate cost effectiveness (performance per cost unit)."""
        performance = self.performance.get_performance_score()
        
        # If no cost data or free API, return raw performance
        if self.cost_profile.estimated_cost == 0 or self.total_requests == 0:
            return performance
        
        # Calculate average cost per request
        avg_cost_per_request = self.cost_profile.estimated_cost / self.total_requests
        
        # Higher score is better - performance divided by cost
        # Scale to a reasonable range
        cost_effectiveness = performance / (avg_cost_per_request * 100)
        
        return min(1.0, cost_effectiveness)  # Cap at 1.0


class APIBackendScheduler:
    """
    Specialized scheduler for API backends in distributed testing.
    
    This scheduler integrates with the distributed testing framework and provides
    API-specific scheduling capabilities including rate limiting, cost optimization,
    and performance-based routing.
    """
    
    def __init__(
        self,
        rate_limit_strategy: APIRateLimitStrategy = APIRateLimitStrategy.STRICT,
        cost_strategy: APICostStrategy = APICostStrategy.BALANCED,
        update_interval: int = 60,
        enable_circuit_breakers: bool = True,
        enable_load_balancing: bool = True,
        config_file: Optional[str] = None
    ):
        """
        Initialize the API backend scheduler.
        
        Args:
            rate_limit_strategy: Strategy for handling API rate limits
            cost_strategy: Strategy for optimizing API costs
            update_interval: Interval in seconds for updating metrics
            enable_circuit_breakers: Whether to enable circuit breakers
            enable_load_balancing: Whether to enable load balancing
            config_file: Optional path to configuration file
        """
        self.rate_limit_strategy = rate_limit_strategy
        self.cost_strategy = cost_strategy
        self.update_interval = update_interval
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_load_balancing = enable_load_balancing
        
        # Load configuration if provided
        self.config = self._load_config(config_file)
        
        # Initialize API backend profiles
        self.api_profiles = {}  # type: Dict[str, APIBackendProfile]
        
        # Initialize scheduling state
        self.task_queue = []  # priority queue
        self.running_tasks = {}  # task_id -> task
        self.pending_tasks = {}  # task_id -> task
        self.completed_tasks = {}  # task_id -> task
        self.failed_tasks = {}  # task_id -> task
        
        # Load balancer for distributing tasks
        self.load_balancer = LoadBalancer()
        
        # Performance metrics
        self.scheduling_metrics = {
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "api_utilization": {},
            "cost_tracking": {},
            "start_time": time.time()
        }
        
        # Start with default API profiles
        self._initialize_default_profiles()
        
        # Locks for thread safety
        self.profile_lock = threading.RLock()
        self.task_lock = threading.RLock()
        
        # Initialize update thread
        self.running = False
        self.update_thread = None
        
        logger.info(f"APIBackendScheduler initialized with {len(self.api_profiles)} API profiles")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "rate_limits": {
                "openai": {
                    "requests_per_minute": 60,
                    "tokens_per_minute": 90000,
                    "concurrent_requests": 50
                },
                "claude": {
                    "requests_per_minute": 40,
                    "tokens_per_minute": 100000,
                    "concurrent_requests": 30
                },
                "groq": {
                    "requests_per_minute": 100,
                    "concurrent_requests": 100
                }
            },
            "costs": {
                "openai": {
                    "gpt-3.5-turbo": {
                        "cost_per_1k_input_tokens": 0.0005,
                        "cost_per_1k_output_tokens": 0.0015
                    },
                    "gpt-4": {
                        "cost_per_1k_input_tokens": 0.01,
                        "cost_per_1k_output_tokens": 0.03
                    }
                },
                "claude": {
                    "claude-3-opus": {
                        "cost_per_1k_input_tokens": 0.015,
                        "cost_per_1k_output_tokens": 0.075
                    },
                    "claude-3-sonnet": {
                        "cost_per_1k_input_tokens": 0.003,
                        "cost_per_1k_output_tokens": 0.015
                    }
                },
                "groq": {
                    "llama3-8b": {
                        "cost_per_1k_input_tokens": 0.0001,
                        "cost_per_1k_output_tokens": 0.0002
                    },
                    "llama3-70b": {
                        "cost_per_1k_input_tokens": 0.0007,
                        "cost_per_1k_output_tokens": 0.0009
                    }
                }
            },
            "capabilities": {
                "openai": ["completion", "chat_completion", "embedding"],
                "claude": ["chat_completion", "embedding"],
                "groq": ["chat_completion"],
                "llvm": ["local_inference"]
            }
        }
        
        # If config file provided, load and merge
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Deep merge
                for section in file_config:
                    if section in default_config and isinstance(default_config[section], dict):
                        default_config[section].update(file_config[section])
                    else:
                        default_config[section] = file_config[section]
                
                logger.info(f"Loaded API scheduler configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return default_config
    
    def _initialize_default_profiles(self):
        """Initialize default API profiles based on configuration."""
        # Initialize profiles for known API types
        api_types = set()
        
        # Add APIs from rate limits
        if "rate_limits" in self.config:
            api_types.update(self.config["rate_limits"].keys())
        
        # Add APIs from costs
        if "costs" in self.config:
            api_types.update(self.config["costs"].keys())
        
        # Add APIs from capabilities
        if "capabilities" in self.config:
            api_types.update(self.config["capabilities"].keys())
        
        # Create profiles for each API type
        for api_type in api_types:
            self.register_api_backend(api_type)
    
    def start(self):
        """Start the API backend scheduler."""
        if self.running:
            logger.warning("API scheduler already running")
            return
        
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("API backend scheduler started")
    
    def stop(self):
        """Stop the API backend scheduler."""
        if not self.running:
            logger.warning("API scheduler not running")
            return
        
        self.running = False
        
        # Wait for thread to stop
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("API backend scheduler stopped")
    
    def _update_loop(self):
        """Background thread for updating metrics and state."""
        while self.running:
            try:
                # Update API profiles
                self._update_api_profiles()
                
                # Update scheduling metrics
                self._update_scheduling_metrics()
                
                # Update load balancer weights
                if self.enable_load_balancing:
                    self._update_load_balancer()
                
            except Exception as e:
                logger.error(f"Error in API scheduler update loop: {e}")
            
            # Sleep until next update
            for _ in range(self.update_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _update_api_profiles(self):
        """Update API backend profiles."""
        with self.profile_lock:
            now = time.time()
            
            for api_type, profile in self.api_profiles.items():
                # Update last_update timestamp
                profile.last_update = now
    
    def _update_scheduling_metrics(self):
        """Update scheduling metrics."""
        with self.task_lock:
            # Update API utilization
            for api_type, profile in self.api_profiles.items():
                self.scheduling_metrics["api_utilization"][api_type] = profile.rate_limits.get_utilization()
            
            # Update cost tracking
            for api_type, profile in self.api_profiles.items():
                self.scheduling_metrics["cost_tracking"][api_type] = profile.cost_profile.get_cost_report()
    
    def _update_load_balancer(self):
        """Update load balancer weights based on API performance."""
        with self.profile_lock:
            # Calculate weights for each API backend
            weights = {}
            
            for api_type, profile in self.api_profiles.items():
                # Start with performance score
                performance = profile.performance.get_performance_score()
                
                # Apply cost effectiveness if using BALANCED or MINIMIZE_COST strategy
                if self.cost_strategy in [APICostStrategy.BALANCED, APICostStrategy.MINIMIZE_COST]:
                    cost_effectiveness = profile.get_cost_effectiveness()
                    
                    if self.cost_strategy == APICostStrategy.BALANCED:
                        # 50/50 mix of performance and cost effectiveness
                        weight = (performance * 0.5) + (cost_effectiveness * 0.5)
                    else:  # MINIMIZE_COST
                        # 20/80 mix of performance and cost effectiveness
                        weight = (performance * 0.2) + (cost_effectiveness * 0.8)
                else:  # PERFORMANCE_FIRST
                    weight = performance
                
                # Apply circuit breaker status
                if not profile.circuit_breaker.allow_request():
                    weight = 0
                
                # Apply rate limit utilization - reduce weight as we approach limits
                utilization = profile.rate_limits.get_utilization()
                if utilization:
                    # Find the highest utilization
                    max_util = max(utilization.values(), default=0)
                    
                    # Apply utilization factor based on rate limit strategy
                    if self.rate_limit_strategy == APIRateLimitStrategy.STRICT:
                        # Strict - linearly reduce weight as utilization increases
                        weight *= (1 - max_util)
                    elif self.rate_limit_strategy == APIRateLimitStrategy.ADAPTIVE:
                        # Adaptive - allow higher utilization if performance is good
                        # Only start reducing at 50% utilization
                        if max_util > 0.5:
                            weight *= (1 - ((max_util - 0.5) * 2))
                    elif self.rate_limit_strategy == APIRateLimitStrategy.BURST_THROTTLE:
                        # Burst-throttle - allow up to 90% utilization with minimal penalty
                        if max_util > 0.9:
                            weight *= (1 - ((max_util - 0.9) * 10))
                
                # Record the final weight
                weights[api_type] = max(0, weight)
            
            # Update load balancer weights
            self.load_balancer.update_weights(weights)
    
    def register_api_backend(self, api_type: str) -> bool:
        """
        Register an API backend for scheduling.
        
        Args:
            api_type: The type of API to register
            
        Returns:
            True if registration was successful
        """
        with self.profile_lock:
            # Skip if already registered
            if api_type in self.api_profiles:
                return True
            
            # Create rate limit profile
            rate_limits = APIRateLimit()
            if "rate_limits" in self.config and api_type in self.config["rate_limits"]:
                rate_limit_config = self.config["rate_limits"][api_type]
                for key, value in rate_limit_config.items():
                    if hasattr(rate_limits, key):
                        setattr(rate_limits, key, value)
            
            # Create cost profile
            cost_profile = APICostProfile()
            
            # Create performance profile
            performance_profile = APIPerformanceProfile()
            
            # Create circuit breaker
            circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                reset_timeout=300,  # 5 minutes
                half_open_timeout=60  # 1 minute
            )
            
            # Initialize capabilities
            capabilities = {}
            if "capabilities" in self.config and api_type in self.config["capabilities"]:
                for capability in self.config["capabilities"][api_type]:
                    capabilities[capability] = True
            
            # Create API backend profile
            profile = APIBackendProfile(
                api_type=api_type,
                rate_limits=rate_limits,
                cost_profile=cost_profile,
                performance=performance_profile,
                circuit_breaker=circuit_breaker,
                capabilities=capabilities
            )
            
            # Register the profile
            self.api_profiles[api_type] = profile
            
            logger.info(f"Registered API backend: {api_type}")
            return True
    
    def update_api_models(self, api_type: str, models: List[str]) -> bool:
        """
        Update the list of available models for an API backend.
        
        Args:
            api_type: The type of API to update
            models: List of available model identifiers
            
        Returns:
            True if update was successful
        """
        with self.profile_lock:
            if api_type not in self.api_profiles:
                if not self.register_api_backend(api_type):
                    return False
            
            # Update models
            self.api_profiles[api_type].models = set(models)
            
            # Update cost profiles for these models
            if "costs" in self.config and api_type in self.config["costs"]:
                for model in models:
                    if model in self.config["costs"][api_type]:
                        # Create a model-specific cost profile
                        model_costs = self.config["costs"][api_type][model]
                        
                        # Store the costs in the API profile
                        # Note: We don't replace the entire cost_profile here,
                        # just update it with model-specific information
                        for key, value in model_costs.items():
                            if hasattr(self.api_profiles[api_type].cost_profile, key):
                                setattr(self.api_profiles[api_type].cost_profile, key, value)
            
            return True
    
    def add_task(self, task: Task) -> bool:
        """
        Add a task to the scheduler.
        
        Args:
            task: The task to add
            
        Returns:
            True if task was added successfully
        """
        with self.task_lock:
            # Check if task already exists
            if task.task_id in self.pending_tasks or task.task_id in self.running_tasks:
                logger.warning(f"Task {task.task_id} already exists")
                return False
            
            # Add to pending tasks
            self.pending_tasks[task.task_id] = task
            
            # Add to priority queue
            heapq.heappush(self.task_queue, (-task.priority, task.submission_time, task.task_id))
            
            logger.debug(f"Added task {task.task_id} with priority {task.priority}")
            return True
    
    def schedule_tasks(self) -> List[Tuple[str, str]]:
        """
        Schedule pending tasks to API backends.
        
        Returns:
            List of (task_id, api_type) assignments
        """
        with self.task_lock, self.profile_lock:
            assignments = []
            
            # Check if we have tasks to schedule
            if not self.task_queue:
                return []
            
            # Get available API backends
            available_apis = self._get_available_apis()
            if not available_apis:
                logger.warning("No API backends available for scheduling")
                return []
            
            # Process tasks in priority order
            while self.task_queue:
                # Peek at the next task
                _, _, task_id = self.task_queue[0]
                
                if task_id not in self.pending_tasks:
                    # Task no longer exists, remove from queue
                    heapq.heappop(self.task_queue)
                    continue
                
                task = self.pending_tasks[task_id]
                
                # Find suitable API backend for this task
                api_type = self._select_api_for_task(task, available_apis)
                if not api_type:
                    # No suitable API found, try next task
                    break
                
                # Remove task from queue
                heapq.heappop(self.task_queue)
                
                # Remove from pending tasks
                del self.pending_tasks[task_id]
                
                # Add to running tasks
                task.start_time = time.time()
                task.assigned_api = api_type
                self.running_tasks[task_id] = task
                
                # Add to assignments
                assignments.append((task_id, api_type))
                
                # Update metrics
                self.scheduling_metrics["tasks_scheduled"] += 1
                
                # Update API profile
                # Estimate tokens if available in task metadata
                input_tokens = task.metadata.get("estimated_input_tokens", 0)
                self.api_profiles[api_type].record_request_start(input_tokens)
                
                logger.debug(f"Scheduled task {task_id} to API backend {api_type}")
                
                # Check if API is still available after this assignment
                if not self._is_api_available(api_type):
                    available_apis.remove(api_type)
                    if not available_apis:
                        break
            
            return assignments
    
    def complete_task(self, task_id: str, success: bool, result: Any = None) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the completed task
            success: Whether the task completed successfully
            result: Result data or error information
            
        Returns:
            True if task was marked as completed
        """
        with self.task_lock, self.profile_lock:
            # Check if task exists and is running
            if task_id not in self.running_tasks:
                logger.warning(f"Attempted to complete non-running task {task_id}")
                return False
            
            # Get the task
            task = self.running_tasks[task_id]
            api_type = task.assigned_api
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - task.start_time if task.start_time else 0
            
            # Remove from running tasks
            del self.running_tasks[task_id]
            
            # Update API profile based on result
            if api_type in self.api_profiles:
                if success:
                    # Extract token information from result if available
                    input_tokens = 0
                    output_tokens = 0
                    
                    if isinstance(result, dict):
                        input_tokens = result.get("input_tokens", 0)
                        output_tokens = result.get("output_tokens", 0)
                    
                    # Fall back to metadata if not in result
                    if input_tokens == 0 and "estimated_input_tokens" in task.metadata:
                        input_tokens = task.metadata["estimated_input_tokens"]
                    
                    if output_tokens == 0 and "estimated_output_tokens" in task.metadata:
                        output_tokens = task.metadata["estimated_output_tokens"]
                    
                    # Record successful completion
                    self.api_profiles[api_type].record_request_success(
                        latency=duration,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
                    
                    # Add to completed tasks
                    task.end_time = end_time
                    task.result = result
                    self.completed_tasks[task_id] = task
                    
                    # Update metrics
                    self.scheduling_metrics["tasks_completed"] += 1
                    
                else:
                    # Extract error information
                    error = str(result) if result else "Unknown error"
                    
                    # Record failure
                    self.api_profiles[api_type].record_request_failure(error)
                    
                    # Add to failed tasks
                    task.end_time = end_time
                    task.error = error
                    self.failed_tasks[task_id] = task
                    
                    # Update metrics
                    self.scheduling_metrics["tasks_failed"] += 1
            
            logger.debug(f"Task {task_id} completed with success={success}")
            return True
    
    def _get_available_apis(self) -> List[str]:
        """Get list of available API backends."""
        available = []
        
        for api_type, profile in self.api_profiles.items():
            if self._is_api_available(api_type):
                available.append(api_type)
        
        return available
    
    def _is_api_available(self, api_type: str) -> bool:
        """Check if an API backend is available for new tasks."""
        if api_type not in self.api_profiles:
            return False
        
        profile = self.api_profiles[api_type]
        
        # Check circuit breaker
        if self.enable_circuit_breakers and not profile.circuit_breaker.allow_request():
            return False
        
        # Check rate limits
        can_handle, _ = profile.can_handle_request()
        if not can_handle:
            return False
        
        return True
    
    def _select_api_for_task(self, task: Task, available_apis: List[str]) -> Optional[str]:
        """
        Select the best API backend for a task.
        
        Args:
            task: The task to schedule
            available_apis: List of available API backends
            
        Returns:
            Selected API type or None if no suitable API found
        """
        # Filter APIs by task requirements
        suitable_apis = []
        
        for api_type in available_apis:
            profile = self.api_profiles[api_type]
            
            # Check if API supports required capabilities
            if "required_capabilities" in task.metadata:
                required = task.metadata["required_capabilities"]
                if not all(cap in profile.capabilities and profile.capabilities[cap] for cap in required):
                    continue
            
            # Check if API supports requested model
            if "model" in task.metadata and task.metadata["model"]:
                if task.metadata["model"] not in profile.models and len(profile.models) > 0:
                    continue
            
            # Check for token limits
            if "estimated_input_tokens" in task.metadata:
                input_tokens = task.metadata["estimated_input_tokens"]
                can_handle, reason = profile.can_handle_request(input_tokens)
                if not can_handle:
                    logger.debug(f"API {api_type} rejected task {task.task_id}: {reason}")
                    continue
            
            # This API is suitable
            suitable_apis.append(api_type)
        
        if not suitable_apis:
            return None
        
        # Select API using load balancer (if enabled)
        if self.enable_load_balancing and len(suitable_apis) > 1:
            return self.load_balancer.select_backend(suitable_apis)
        
        # Otherwise, pick the first one
        return suitable_apis[0]
    
    def get_api_status(self, api_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status information for API backends.
        
        Args:
            api_type: Optional specific API to get status for
            
        Returns:
            Dictionary of API status information
        """
        with self.profile_lock:
            if api_type:
                if api_type not in self.api_profiles:
                    return {
                        "error": f"API backend '{api_type}' not found"
                    }
                
                return self.api_profiles[api_type].get_health_status()
            
            # Return status for all APIs
            return {
                api: profile.get_health_status()
                for api, profile in self.api_profiles.items()
            }
    
    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """Get scheduling metrics."""
        with self.task_lock, self.profile_lock:
            metrics = self.scheduling_metrics.copy()
            
            # Add current queue lengths
            metrics["pending_tasks"] = len(self.pending_tasks)
            metrics["running_tasks"] = len(self.running_tasks)
            metrics["completed_tasks"] = len(self.completed_tasks)
            metrics["failed_tasks"] = len(self.failed_tasks)
            
            # Add API performance metrics
            metrics["api_performance"] = {
                api: {
                    "avg_latency": profile.performance.get_avg_latency(),
                    "success_rate": profile.get_success_rate(),
                    "performance_score": profile.performance.get_performance_score()
                }
                for api, profile in self.api_profiles.items()
            }
            
            # Add load balancer weights
            metrics["load_balancer_weights"] = self.load_balancer.get_weights()
            
            return metrics
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get a cost report for all API usage."""
        with self.profile_lock:
            report = {
                "total_cost": 0.0,
                "api_costs": {},
                "cost_by_model": {}
            }
            
            # Calculate costs by API
            for api_type, profile in self.api_profiles.items():
                api_cost = profile.cost_profile.estimated_cost
                report["api_costs"][api_type] = api_cost
                report["total_cost"] += api_cost
            
            # TODO: Add cost by model when we track this information
            
            return report
    
    def reset_usage_tracking(self) -> None:
        """Reset all usage tracking data."""
        with self.profile_lock, self.task_lock:
            # Reset API profiles
            for profile in self.api_profiles.values():
                # Reset cost tracking
                profile.cost_profile.total_input_tokens = 0
                profile.cost_profile.total_output_tokens = 0
                profile.cost_profile.total_requests = 0
                profile.cost_profile.estimated_cost = 0.0
                
                # Reset request tracking
                profile.total_requests = 0
                profile.successful_requests = 0
                profile.failed_requests = 0
            
            # Reset scheduling metrics
            self.scheduling_metrics["tasks_scheduled"] = 0
            self.scheduling_metrics["tasks_completed"] = 0
            self.scheduling_metrics["tasks_failed"] = 0
            self.scheduling_metrics["start_time"] = time.time()
            
            logger.info("Reset all usage tracking data")


if __name__ == "__main__":
    # Example usage
    scheduler = APIBackendScheduler()
    scheduler.start()
    
    try:
        # Register some API backends
        scheduler.register_api_backend("openai")
        scheduler.register_api_backend("claude")
        scheduler.register_api_backend("groq")
        
        # Update available models
        scheduler.update_api_models("openai", ["gpt-3.5-turbo", "gpt-4"])
        scheduler.update_api_models("claude", ["claude-3-opus", "claude-3-sonnet"])
        scheduler.update_api_models("groq", ["llama3-8b", "llama3-70b"])
        
        # Create some tasks
        for i in range(10):
            task = Task(
                task_id=f"task-{i}",
                task_type="api_test",
                user_id="test_user",
                priority=random.randint(1, 10),
                required_resources={},
                metadata={
                    "test_type": "latency",
                    "model": "gpt-3.5-turbo" if i % 3 == 0 else "claude-3-sonnet" if i % 3 == 1 else "llama3-8b",
                    "estimated_input_tokens": random.randint(100, 1000),
                    "estimated_output_tokens": random.randint(50, 500),
                    "required_capabilities": ["chat_completion"]
                },
                submission_time=time.time(),
                dependencies=[]
            )
            scheduler.add_task(task)
        
        # Schedule tasks
        assignments = scheduler.schedule_tasks()
        print(f"Scheduled {len(assignments)} tasks:")
        for task_id, api_type in assignments:
            print(f"  Task {task_id} -> {api_type}")
        
        # Simulate task completion
        for task_id, api_type in assignments:
            success = random.random() > 0.2  # 80% success rate
            result = {
                "success": success,
                "latency": random.uniform(0.2, 2.0),
                "input_tokens": random.randint(100, 1000),
                "output_tokens": random.randint(50, 500)
            } if success else f"Error: API timeout"
            
            scheduler.complete_task(task_id, success, result)
        
        # Get API status
        print("\nAPI Status:")
        status = scheduler.get_api_status()
        for api_type, api_status in status.items():
            print(f"  {api_type}: Circuit breaker = {api_status['circuit_breaker_status']}, "
                  f"Success rate = {api_status['success_rate']:.2f}, "
                  f"Performance score = {api_status['performance_score']:.2f}")
        
        # Get scheduling metrics
        print("\nScheduling Metrics:")
        metrics = scheduler.get_scheduling_metrics()
        print(f"  Tasks: {metrics['tasks_completed']} completed, {metrics['tasks_failed']} failed")
        
        # Get cost report
        print("\nCost Report:")
        cost_report = scheduler.get_cost_report()
        print(f"  Total cost: ${cost_report['total_cost']:.4f}")
        for api, cost in cost_report['api_costs'].items():
            print(f"  {api}: ${cost:.4f}")
        
    finally:
        # Stop the scheduler
        scheduler.stop()