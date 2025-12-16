"""
Timing utilities for NAMI performance profiling

Simple, lightweight timing to identify bottlenecks
"""

import time
from contextlib import contextmanager
from collections import defaultdict
import threading


class Timer:
    """Thread-safe timer for profiling NAMI operations"""

    def __init__(self):
        self.timings = defaultdict(lambda: {'count': 0, 'total': 0.0, 'min': float('inf'), 'max': 0.0})
        self.lock = threading.Lock()

    @contextmanager
    def measure(self, operation_name):
        """Context manager to time an operation"""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            with self.lock:
                stats = self.timings[operation_name]
                stats['count'] += 1
                stats['total'] += elapsed
                stats['min'] = min(stats['min'], elapsed)
                stats['max'] = max(stats['max'], elapsed)

    def get_report(self):
        """Generate timing report"""
        if not self.timings:
            return "No timing data collected"

        lines = []
        lines.append("\n" + "="*80)
        lines.append("TIMING REPORT")
        lines.append("="*80)

        # Calculate total time
        total_time = sum(stats['total'] for stats in self.timings.values())

        # Sort by total time (descending)
        sorted_timings = sorted(self.timings.items(),
                              key=lambda x: x[1]['total'],
                              reverse=True)

        lines.append(f"{'Operation':<40} {'Count':>8} {'Total(s)':>12} {'Avg(s)':>12} {'Min(s)':>12} {'Max(s)':>12} {'%':>8}")
        lines.append("-"*80)

        for operation, stats in sorted_timings:
            count = stats['count']
            total = stats['total']
            avg = total / count if count > 0 else 0
            min_time = stats['min'] if stats['min'] != float('inf') else 0
            max_time = stats['max']
            percent = (total / total_time * 100) if total_time > 0 else 0

            lines.append(f"{operation:<40} {count:>8} {total:>12.3f} {avg:>12.3f} {min_time:>12.3f} {max_time:>12.3f} {percent:>7.1f}%")

        lines.append("-"*80)
        lines.append(f"{'TOTAL':<40} {sum(s['count'] for s in self.timings.values()):>8} {total_time:>12.3f}")
        lines.append("="*80)

        return "\n".join(lines)

    def print_report(self):
        """Print timing report to stdout"""
        print(self.get_report())

    def save_report(self, filename):
        """Save timing report to file"""
        with open(filename, 'w') as f:
            f.write(self.get_report())


# Global timer instance
_global_timer = Timer()


def get_timer():
    """Get the global timer instance"""
    return _global_timer


@contextmanager
def time_operation(operation_name):
    """Convenience function for timing operations"""
    with _global_timer.measure(operation_name):
        yield
