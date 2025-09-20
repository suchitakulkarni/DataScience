import time
import psutil
import logging
from functools import wraps
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PipelineMetrics:
    """Track pipeline metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing and record duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[f"{operation}_duration"] = duration
            del self.start_times[operation]
            logger.info(f"{operation} completed in {duration:.2f}s")
    
    def record_metric(self, name: str, value: Any):
        """Record a metric value"""
        self.metrics[name] = value
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return {
            'memory_usage_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent()
        }

def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = getattr(args[0], 'metrics', PipelineMetrics())
            metrics.start_timer(operation_name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                metrics.end_timer(operation_name)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = getattr(args[0], 'metrics', PipelineMetrics())
            metrics.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics.end_timer(operation_name)
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
