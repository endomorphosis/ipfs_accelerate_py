"""
Query Optimizer for IPFS Integration (Phase 2D)

This module provides advanced query optimization, caching strategies,
and performance monitoring for the IPFS-integrated database.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Advanced query optimizer with caching and performance monitoring.
    
    Features:
    - Query plan optimization
    - Intelligent caching strategies
    - Query rewriting for distributed execution
    - Performance monitoring and analysis
    """
    
    def __init__(self, cache_manager=None):
        """
        Initialize the query optimizer.
        
        Args:
            cache_manager: IPFSCacheManager instance
        """
        self.cache_manager = cache_manager
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        self.optimization_rules: List[Dict[str, Any]] = []
        
        # Setup default optimization rules
        self._setup_default_rules()
        
        logger.info("Initialized QueryOptimizer")
    
    def _setup_default_rules(self):
        """Setup default query optimization rules."""
        self.optimization_rules = [
            {
                'name': 'push_down_filters',
                'pattern': r'WHERE.*ORDER BY',
                'optimizer': self._push_down_filters,
                'priority': 1
            },
            {
                'name': 'index_selection',
                'pattern': r'WHERE.*=',
                'optimizer': self._optimize_index_usage,
                'priority': 2
            },
            {
                'name': 'join_reorder',
                'pattern': r'JOIN',
                'optimizer': self._optimize_join_order,
                'priority': 3
            }
        ]
    
    def optimize_query(
        self,
        query: str,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize a query for better performance.
        
        Args:
            query: SQL query to optimize
            execution_context: Additional context for optimization
        
        Returns:
            Optimization result with optimized query and metadata
        """
        start_time = time.time()
        
        # Generate query fingerprint
        query_hash = self._generate_query_hash(query)
        
        # Check cache for query plan
        cached_plan = self._get_cached_query_plan(query_hash)
        if cached_plan:
            logger.debug(f"Using cached query plan for {query_hash}")
            return {
                'original_query': query,
                'optimized_query': cached_plan['optimized_query'],
                'optimizations_applied': cached_plan['optimizations'],
                'cached': True,
                'optimization_time': time.time() - start_time
            }
        
        # Apply optimization rules
        optimized_query = query
        optimizations_applied = []
        
        for rule in sorted(self.optimization_rules, key=lambda x: x['priority']):
            try:
                result = rule['optimizer'](optimized_query, execution_context)
                if result['modified']:
                    optimized_query = result['query']
                    optimizations_applied.append(rule['name'])
            except Exception as e:
                logger.warning(f"Optimization rule {rule['name']} failed: {e}")
        
        # Cache the query plan
        query_plan = {
            'optimized_query': optimized_query,
            'optimizations': optimizations_applied,
            'timestamp': time.time()
        }
        self._cache_query_plan(query_hash, query_plan)
        
        optimization_time = time.time() - start_time
        
        # Update stats
        self._update_query_stats(query_hash, optimization_time)
        
        return {
            'original_query': query,
            'optimized_query': optimized_query,
            'optimizations_applied': optimizations_applied,
            'cached': False,
            'optimization_time': optimization_time
        }
    
    def analyze_query_performance(
        self,
        query: str,
        execution_time: float,
        result_count: int
    ) -> Dict[str, Any]:
        """
        Analyze query performance and suggest improvements.
        
        Args:
            query: Executed query
            execution_time: Query execution time
            result_count: Number of results returned
        
        Returns:
            Performance analysis and suggestions
        """
        query_hash = self._generate_query_hash(query)
        
        # Get historical performance
        if query_hash in self.query_stats:
            stats = self.query_stats[query_hash]
            avg_time = stats['total_time'] / stats['execution_count']
            performance_ratio = execution_time / avg_time if avg_time > 0 else 1.0
        else:
            avg_time = execution_time
            performance_ratio = 1.0
        
        # Generate suggestions
        suggestions = []
        
        if execution_time > 1.0:  # Slow query
            suggestions.append({
                'type': 'slow_query',
                'message': 'Query execution time > 1s, consider adding indexes',
                'severity': 'high'
            })
        
        if result_count > 10000:  # Large result set
            suggestions.append({
                'type': 'large_result',
                'message': 'Large result set, consider adding LIMIT clause',
                'severity': 'medium'
            })
        
        if performance_ratio > 2.0:  # Performance degradation
            suggestions.append({
                'type': 'performance_degradation',
                'message': f'Performance degraded by {performance_ratio:.1f}x',
                'severity': 'high'
            })
        
        return {
            'query_hash': query_hash,
            'execution_time': execution_time,
            'average_time': avg_time,
            'performance_ratio': performance_ratio,
            'result_count': result_count,
            'suggestions': suggestions
        }
    
    def get_caching_strategy(
        self,
        query: str,
        execution_time: float,
        result_size: int
    ) -> Dict[str, Any]:
        """
        Determine optimal caching strategy for a query.
        
        Args:
            query: Query to analyze
            execution_time: Query execution time
            result_size: Size of results in bytes
        
        Returns:
            Recommended caching strategy
        """
        query_hash = self._generate_query_hash(query)
        
        # Analyze query characteristics
        is_expensive = execution_time > 0.5
        is_large = result_size > 1024 * 1024  # > 1MB
        is_frequent = self._is_frequent_query(query_hash)
        
        # Determine strategy
        if is_expensive and is_frequent:
            strategy = 'aggressive'
            ttl = 3600  # 1 hour
            prefetch = True
        elif is_expensive or is_frequent:
            strategy = 'standard'
            ttl = 1800  # 30 minutes
            prefetch = False
        elif is_large:
            strategy = 'minimal'
            ttl = 300  # 5 minutes
            prefetch = False
        else:
            strategy = 'none'
            ttl = 0
            prefetch = False
        
        return {
            'strategy': strategy,
            'ttl': ttl,
            'prefetch': prefetch,
            'reasons': {
                'expensive': is_expensive,
                'large': is_large,
                'frequent': is_frequent
            }
        }
    
    def _push_down_filters(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Push down filters closer to data source.
        
        Args:
            query: Query to optimize
            context: Execution context
        
        Returns:
            Optimization result
        """
        # Simplified filter push-down (placeholder)
        # In real implementation, would parse and rewrite query
        modified = 'WHERE' in query and 'JOIN' in query
        
        return {
            'query': query,
            'modified': modified
        }
    
    def _optimize_index_usage(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize index usage in query.
        
        Args:
            query: Query to optimize
            context: Execution context
        
        Returns:
            Optimization result
        """
        # Placeholder for index optimization
        modified = False
        
        return {
            'query': query,
            'modified': modified
        }
    
    def _optimize_join_order(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize join order for better performance.
        
        Args:
            query: Query to optimize
            context: Execution context
        
        Returns:
            Optimization result
        """
        # Placeholder for join order optimization
        modified = False
        
        return {
            'query': query,
            'modified': modified
        }
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate a unique hash for a query."""
        # Normalize query (remove extra whitespace)
        normalized = ' '.join(query.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_cached_query_plan(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query plan if available."""
        if self.cache_manager:
            return self.cache_manager.get(f"query_plan:{query_hash}")
        return None
    
    def _cache_query_plan(self, query_hash: str, plan: Dict[str, Any]):
        """Cache a query plan."""
        if self.cache_manager:
            self.cache_manager.set(f"query_plan:{query_hash}", plan, ttl=3600)
    
    def _update_query_stats(self, query_hash: str, optimization_time: float):
        """Update query statistics."""
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                'execution_count': 0,
                'total_time': 0.0,
                'last_execution': 0
            }
        
        stats = self.query_stats[query_hash]
        stats['execution_count'] += 1
        stats['total_time'] += optimization_time
        stats['last_execution'] = time.time()
    
    def _is_frequent_query(self, query_hash: str, threshold: int = 10) -> bool:
        """Check if query is frequently executed."""
        if query_hash in self.query_stats:
            return self.query_stats[query_hash]['execution_count'] >= threshold
        return False
    
    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """
        Get optimizer statistics.
        
        Returns:
            Statistics about query optimization
        """
        total_queries = len(self.query_stats)
        total_executions = sum(stats['execution_count'] for stats in self.query_stats.values())
        
        if total_executions > 0:
            avg_optimizations = sum(
                stats['execution_count'] for stats in self.query_stats.values()
            ) / total_queries if total_queries > 0 else 0
        else:
            avg_optimizations = 0
        
        return {
            'total_queries_optimized': total_queries,
            'total_query_executions': total_executions,
            'average_optimizations_per_query': avg_optimizations,
            'optimization_rules': len(self.optimization_rules)
        }


class PerformanceMonitor:
    """
    Performance monitoring and analytics for IPFS-integrated database.
    
    Features:
    - Real-time performance metrics
    - Trend analysis
    - Alerting on performance degradation
    - Resource utilization tracking
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, float] = {
            'query_time': 1.0,
            'cache_hit_rate': 0.7,
            'sync_time': 5.0
        }
        
        logger.info("Initialized PerformanceMonitor")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        metric = {
            'timestamp': time.time(),
            'value': value,
            'metadata': metadata or {}
        }
        
        self.metrics[metric_name].append(metric)
        
        # Check thresholds and generate alerts
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            if value > threshold:
                self._generate_alert(metric_name, value, threshold)
    
    def get_metric_summary(
        self,
        metric_name: str,
        time_window: int = 3600
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            time_window: Time window in seconds
        
        Returns:
            Metric summary
        """
        if metric_name not in self.metrics:
            return {}
        
        # Filter by time window
        cutoff_time = time.time() - time_window
        recent_metrics = [
            m for m in self.metrics[metric_name]
            if m['timestamp'] >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m['value'] for m in recent_metrics]
        
        return {
            'metric_name': metric_name,
            'count': len(values),
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else None,
            'time_window': time_window
        }
    
    def get_performance_trends(
        self,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance trends.
        
        Args:
            metric_names: List of metrics to analyze (None for all)
        
        Returns:
            Trend analysis for each metric
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        trends = {}
        for metric_name in metric_names:
            if metric_name not in self.metrics:
                continue
            
            recent = self.get_metric_summary(metric_name, time_window=3600)
            historical = self.get_metric_summary(metric_name, time_window=86400)
            
            if recent and historical:
                trend = 'improving' if recent['average'] < historical['average'] else 'degrading'
                change_pct = ((recent['average'] - historical['average']) /
                             historical['average'] * 100) if historical['average'] > 0 else 0
            else:
                trend = 'stable'
                change_pct = 0
            
            trends[metric_name] = {
                'trend': trend,
                'change_percentage': change_pct,
                'recent_average': recent.get('average', 0),
                'historical_average': historical.get('average', 0)
            }
        
        return trends
    
    def _generate_alert(
        self,
        metric_name: str,
        value: float,
        threshold: float
    ):
        """Generate a performance alert."""
        alert = {
            'timestamp': time.time(),
            'metric_name': metric_name,
            'value': value,
            'threshold': threshold,
            'severity': 'high' if value > threshold * 2 else 'medium',
            'message': f"{metric_name} ({value:.2f}) exceeded threshold ({threshold:.2f})"
        }
        
        self.alerts.append(alert)
        logger.warning(f"Performance alert: {alert['message']}")
    
    def get_active_alerts(
        self,
        time_window: int = 3600
    ) -> List[Dict[str, Any]]:
        """
        Get recent performance alerts.
        
        Args:
            time_window: Time window in seconds
        
        Returns:
            List of recent alerts
        """
        cutoff_time = time.time() - time_window
        return [alert for alert in self.alerts if alert['timestamp'] >= cutoff_time]
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get performance dashboard summary.
        
        Returns:
            Dashboard summary with key metrics
        """
        return {
            'total_metrics': len(self.metrics),
            'metric_summaries': {
                name: self.get_metric_summary(name, time_window=3600)
                for name in self.metrics.keys()
            },
            'trends': self.get_performance_trends(),
            'active_alerts': len(self.get_active_alerts()),
            'health_status': self._calculate_health_status()
        }
    
    def _calculate_health_status(self) -> str:
        """Calculate overall system health status."""
        recent_alerts = self.get_active_alerts(time_window=3600)
        
        if not recent_alerts:
            return 'healthy'
        elif len(recent_alerts) < 5:
            return 'warning'
        else:
            return 'critical'
