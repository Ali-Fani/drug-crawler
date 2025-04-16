import json
import os
import time
from typing import Any, Dict

# Performance monitoring
class PerformanceMonitor:
    """Track performance metrics for the crawler"""
    
    def __init__(self):
        self.start_time = time.time()
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.request_times = []
        
    def add_request_time(self, elapsed: float, success: bool = True):
        """Add a request time to the monitor"""
        self.requests_total += 1
        self.request_times.append(elapsed)
        if success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the current performance statistics"""
        if not self.request_times:
            return {
                "total_time": time.time() - self.start_time,
                "requests_total": 0,
                "requests_success": 0,
                "requests_failed": 0,
                "avg_request_time": 0,
                "min_request_time": 0,
                "max_request_time": 0
            }
            
        return {
            "total_time": time.time() - self.start_time,
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "avg_request_time": sum(self.request_times) / len(self.request_times),
            "min_request_time": min(self.request_times),
            "max_request_time": max(self.request_times)
        }
    
    def print_summary(self):
        """Print a summary of the performance metrics"""
        stats = self.get_stats()
        
        print("\n📊 Performance Summary:")
        print(f"  • Total Runtime: {stats['total_time']:.2f} seconds")
        print(f"  • Total Requests: {stats['requests_total']}")
        print(f"  • Successful Requests: {stats['requests_success']}")
        print(f"  • Failed Requests: {stats['requests_failed']}")
        if stats['requests_total'] > 0:
            print(f"  • Success Rate: {stats['requests_success']/stats['requests_total']*100:.2f}%")
            print(f"  • Average Request Time: {stats['avg_request_time']:.4f} seconds")
            print(f"  • Fastest Request: {stats['min_request_time']:.4f} seconds")
            print(f"  • Slowest Request: {stats['max_request_time']:.4f} seconds")
        
# Batch processing helpers
async def process_in_batches(items, batch_size, process_func, *args, **kwargs):
    """Process items in batches with progress tracking"""
    results = []
    total_items = len(items)
    
    for i in range(0, total_items, batch_size):
        batch = items[i:i+batch_size]
        batch_results = await process_func(batch, *args, **kwargs)
        results.extend(batch_results)
        
    return results

# Cache management
class RequestCache:
    """Simple cache for request results"""
    
    def __init__(self, cache_dir: str, max_age_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_age_seconds = max_age_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from a URL"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    async def get(self, url: str) -> Any:
        """Get a cached response or None if not found/expired"""
        key = self._get_cache_key(url)
        path = self._get_cache_path(key)
        
        if not os.path.exists(path):
            return None
        
        # Check if cache is expired
        if time.time() - os.path.getmtime(path) > self.max_age_seconds:
            return None
        try:
            import aiofiles
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception:
            return None
            
    async def set(self, url: str, data: Any) -> None:
        """Cache response data for a URL"""
        key = self._get_cache_key(url)
        path = self._get_cache_path(key)
        
        try:
            import aiofiles
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f"Cache error for {url}: {str(e)}")