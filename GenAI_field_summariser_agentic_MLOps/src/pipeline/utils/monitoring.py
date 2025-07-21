# src/pipeline/utils/monitoring.py
"""
Monitoring and metrics collection
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import psutil
import threading


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores pipeline metrics"""
    
    def __init__(self):
        self.metrics: List[MetricPoint] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self._lock:
            metric = MetricPoint(
                name=name,
                value=value,
                tags=tags or {}
            )
            self.metrics.append(metric)
            self.logger.debug(f"Recorded metric: {name}={value} {tags}")
    
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[MetricPoint]:
        """Retrieve metrics with optional filtering"""
        with self._lock:
            filtered = self.metrics
            
            if name:
                filtered = [m for m in filtered if m.name == name]
            
            if since:
                filtered = [m for m in filtered if m.timestamp >= since]
            
            return filtered.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics"""
        with self._lock:
            if not self.metrics:
                return {"total_metrics": 0}
            
            summary = {
                "total_metrics": len(self.metrics),
                "metric_types": len(set(m.name for m in self.metrics)),
                "time_range": {
                    "start": min(m.timestamp for m in self.metrics),
                    "end": max(m.timestamp for m in self.metrics)
                }
            }
            
            # Aggregate by metric name
            by_name = {}
            for metric in self.metrics:
                if metric.name not in by_name:
                    by_name[metric.name] = []
                by_name[metric.name].append(metric.value)
            
            summary["by_metric"] = {}
            for name, values in by_name.items():
                summary["by_metric"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
            
            return summary
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        with self._lock:
            self.metrics.clear()


class PipelineMonitor:
    """Monitors pipeline execution with metrics and system resources"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger(__name__)
        self._start_time = None
    
    @contextmanager
    def monitor_operation(self, operation_name: str, **tags):
        """Context manager to monitor operation duration and resources"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        self.logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield
            
            # Record success metrics
            duration = time.time() - start_time
            memory_delta = self._get_memory_usage() - start_memory
            
            self.metrics.record_metric(
                f"{operation_name}.duration", 
                duration,
                {**tags, "status": "success"}
            )
            self.metrics.record_metric(
                f"{operation_name}.memory_delta", 
                memory_delta,
                {**tags, "status": "success"}
            )
            
            self.logger.info(
                f"Completed operation: {operation_name} "
                f"(duration={duration:.2f}s, memory_delta={memory_delta:.1f}MB)"
            )
            
        except Exception as e:
            # Record failure metrics
            duration = time.time() - start_time
            
            self.metrics.record_metric(
                f"{operation_name}.duration", 
                duration,
                {**tags, "status": "error"}
            )
            self.metrics.record_metric(
                f"{operation_name}.errors", 
                1,
                {**tags, "error_type": type(e).__name__}
            )
            
            self.logger.error(
                f"Failed operation: {operation_name} "
                f"(duration={duration:.2f}s, error={str(e)})"
            )
            
            raise
    
    def record_pipeline_start(self):
        """Record pipeline start"""
        self._start_time = time.time()
        self.metrics.record_metric("pipeline.started", 1)
        self.logger.info("Pipeline execution started")
    
    def record_pipeline_end(self, success: bool = True):
        """Record pipeline completion"""
        if self._start_time:
            total_duration = time.time() - self._start_time
            self.metrics.record_metric(
                "pipeline.total_duration", 
                total_duration,
                {"status": "success" if success else "error"}
            )
        
        self.metrics.record_metric(
            "pipeline.completed", 
            1,
            {"status": "success" if success else "error"}
        )
        
        status = "successfully" if success else "with errors"
        self.logger.info(f"Pipeline execution completed {status}")
    
    def record_data_point(self, name: str, value: float, **tags):
        """Record a custom data point"""
        self.metrics.record_metric(name, value, tags)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information"""
        return {
            "memory_usage_mb": self._get_memory_usage(),
            "cpu_percent": psutil.cpu_percent(),
            "disk_usage": dict(psutil.disk_usage('/')),
            "timestamp": datetime.now()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
