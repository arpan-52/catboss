#!/usr/bin/env python3
"""
CATBOSS POOH Performance Profiler
==================================

Professional profiling script for analyzing POOH performance on real datasets.
Measures execution time, memory usage, and identifies bottlenecks.

Usage:
    python profile_pooh.py <ms_file> [options]

Example:
    python profile_pooh.py data.ms --sigma 6.0 --rho 1.5

Output:
    - Detailed timing breakdown by component
    - Memory usage statistics
    - Throughput metrics
    - Performance bottleneck identification
"""

import argparse
import sys
import time
import os
import cProfile
import pstats
import io
from pstats import SortKey
import psutil
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from catboss.pooh import pooh


class PerformanceProfiler:
    """Professional performance profiler for POOH execution"""

    def __init__(self):
        self.timings = {}
        self.memory_samples = []
        self.process = psutil.Process()

    def start_timing(self, component):
        """Start timing a component"""
        self.timings[component] = {"start": time.time()}

    def end_timing(self, component):
        """End timing a component"""
        if component in self.timings and "start" in self.timings[component]:
            elapsed = time.time() - self.timings[component]["start"]
            self.timings[component]["duration"] = elapsed
            return elapsed
        return 0

    def sample_memory(self):
        """Sample current memory usage"""
        mem_info = self.process.memory_info()
        self.memory_samples.append(
            {"rss_gb": mem_info.rss / 1e9, "timestamp": time.time()}
        )

    def print_report(self, results, total_time):
        """Print professional performance report"""
        print("\n" + "=" * 80)
        print(" CATBOSS POOH PERFORMANCE REPORT")
        print("=" * 80)

        # Execution Summary
        print("\n[EXECUTION SUMMARY]")
        print(f"  Total Runtime:              {total_time:.2f} seconds")
        print(f"  Baselines Processed:        {results.get('baselines_processed', 0)}")
        print(
            f"  Baselines Skipped:          {results.get('baselines_skipped', 0)}"
        )
        print(
            f"  Total Visibilities:         {results.get('total_visibilities', 0):,}"
        )

        # Throughput
        vis_per_sec = results.get("total_visibilities", 0) / total_time
        print(f"  Throughput:                 {vis_per_sec:,.0f} vis/sec")

        # Flagging Statistics
        print("\n[FLAGGING STATISTICS]")
        print(
            f"  Existing Flags:             {results.get('existing_flags', 0):,} ({results.get('overall_percent_flagged', 0) - results.get('new_percent_flagged', 0):.2f}%)"
        )
        print(
            f"  New POOH Flags:             {results.get('new_flags', 0):,} ({results.get('new_percent_flagged', 0):.2f}%)"
        )
        print(
            f"  Total Flags:                {results.get('existing_flags', 0) + results.get('new_flags', 0):,} ({results.get('overall_percent_flagged', 0):.2f}%)"
        )

        # Memory Usage
        if self.memory_samples:
            peak_mem = max(s["rss_gb"] for s in self.memory_samples)
            avg_mem = np.mean([s["rss_gb"] for s in self.memory_samples])
            print("\n[MEMORY USAGE]")
            print(f"  Peak Memory:                {peak_mem:.2f} GB")
            print(f"  Average Memory:             {avg_mem:.2f} GB")

        # Component Timing Breakdown
        if self.timings:
            print("\n[TIMING BREAKDOWN]")
            sorted_timings = sorted(
                [
                    (k, v.get("duration", 0))
                    for k, v in self.timings.items()
                    if "duration" in v
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            for component, duration in sorted_timings:
                percent = 100 * duration / total_time if total_time > 0 else 0
                print(f"  {component:30s} {duration:8.2f}s ({percent:5.1f}%)")

        # Performance Classification
        print("\n[PERFORMANCE CLASSIFICATION]")
        if total_time < 60:
            rating = "EXCELLENT"
        elif total_time < 300:
            rating = "GOOD"
        elif total_time < 600:
            rating = "ACCEPTABLE"
        else:
            rating = "NEEDS OPTIMIZATION"

        print(f"  Overall Rating:             {rating}")

        print("\n" + "=" * 80 + "\n")


def run_profiled_pooh(ms_file, options, enable_cprofile=False):
    """
    Run POOH with comprehensive profiling

    Args:
        ms_file: Path to Measurement Set
        options: POOH configuration options
        enable_cprofile: Enable detailed cProfile profiling (slower)

    Returns:
        tuple: (results, profiler, profile_stats)
    """
    profiler = PerformanceProfiler()

    print(f"\n[PROFILER] Starting profiling run on {ms_file}")
    print(f"[PROFILER] Configuration: sigma={options.get('sigma_factor')}, rho={options.get('rho')}")
    print(f"[PROFILER] Detailed profiling: {'ENABLED' if enable_cprofile else 'DISABLED'}\n")

    # Start memory monitoring in background
    import threading

    def monitor_memory():
        while getattr(monitor_memory, "running", True):
            profiler.sample_memory()
            time.sleep(1.0)

    monitor_memory.running = True
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    # Run with optional cProfile
    profile_stats = None
    start_time = time.time()

    try:
        if enable_cprofile:
            pr = cProfile.Profile()
            pr.enable()
            results = pooh.hunt_ms(ms_file, options)
            pr.disable()

            # Capture profile stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
            ps.print_stats(30)  # Top 30 functions
            profile_stats = s.getvalue()
        else:
            results = pooh.hunt_ms(ms_file, options)

    finally:
        monitor_memory.running = False
        monitor_thread.join(timeout=2)

    total_time = time.time() - start_time

    # Print report
    profiler.print_report(results, total_time)

    # Print detailed cProfile stats if available
    if profile_stats and enable_cprofile:
        print("\n[DETAILED PROFILE] Top 30 Functions by Cumulative Time")
        print("=" * 80)
        print(profile_stats)

    return results, profiler, profile_stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CATBOSS POOH Performance Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile_pooh.py data.ms
  python profile_pooh.py data.ms --sigma 6.0 --detailed
  python profile_pooh.py data.ms --config config.json --detailed
        """,
    )

    # Required arguments
    parser.add_argument("ms_path", help="Path to Measurement Set file")

    # Optional arguments
    parser.add_argument(
        "--combinations",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Window sizes (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument(
        "--sigma", type=float, default=6.0, help="Sigma factor (default: 6.0)"
    )
    parser.add_argument(
        "--rho", type=float, default=1.5, help="Rho factor (default: 1.5)"
    )
    parser.add_argument(
        "--poly-degree", type=int, default=5, help="Polynomial degree (default: 5)"
    )
    parser.add_argument(
        "--deviation-threshold",
        type=float,
        default=3.0,
        help="Deviation threshold (default: 3.0)",
    )
    parser.add_argument(
        "--polarizations",
        type=str,
        default=None,
        help="Polarization indices (e.g., 0,3)",
    )
    parser.add_argument(
        "--apply-flags",
        action="store_true",
        help="Apply flags to MS (default: False)",
    )
    parser.add_argument(
        "--diagnostic-plots",
        action="store_true",
        help="Generate diagnostic plots (default: False)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--max-threads", type=int, default=16, help="Max threads (default: 16)"
    )
    parser.add_argument(
        "--max-memory-usage",
        type=float,
        default=0.8,
        help="Max memory fraction (default: 0.8)",
    )
    parser.add_argument(
        "--config", type=str, help="Path to JSON config file (overrides CLI args)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Enable detailed cProfile profiling (slower)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load config or build from CLI args
    if args.config:
        import json

        with open(args.config, "r") as f:
            options = json.load(f)
    else:
        # Parse combinations
        combinations = [int(x.strip()) for x in args.combinations.split(",")]

        # Parse polarizations
        corr_to_process = None
        if args.polarizations:
            corr_to_process = [int(x.strip()) for x in args.polarizations.split(",")]

        options = {
            "combinations": combinations,
            "sigma_factor": args.sigma,
            "rho": args.rho,
            "poly_degree": args.poly_degree,
            "deviation_threshold": args.deviation_threshold,
            "corr_to_process": corr_to_process,
            "apply_flags": args.apply_flags,
            "diagnostic_plots": args.diagnostic_plots,
            "output_dir": args.output_dir,
            "max_threads": args.max_threads,
            "max_memory_usage": args.max_memory_usage,
            "verbose": args.verbose,
        }

    # Run profiled execution
    results, profiler, profile_stats = run_profiled_pooh(
        args.ms_path, options, enable_cprofile=args.detailed
    )

    print(f"[PROFILER] Profiling complete. Results stored in memory.")
    print(f"[PROFILER] Run with --detailed for cProfile breakdown.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
