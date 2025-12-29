#!/usr/bin/env python
"""
Benchmark different MS reading approaches to find the fastest for HDD.
Usage: python benchmark_ms_reads.py <ms_file>
"""

import sys
import time
import numpy as np
from daskms import xds_from_ms
import dask
from casacore.tables import table

def benchmark_daskms_or_query(ms_file, field_id, baselines, chunk_size):
    """Current approach: daskms with complex OR query"""
    print("\n1. Testing daskms with OR query (CURRENT APPROACH)...")

    # Build OR query like current code
    baseline_clauses = [f"(ANTENNA1={bl[0]} AND ANTENNA2={bl[1]})" for bl in baselines]
    taql_where = f"FIELD_ID={field_id} AND ({' OR '.join(baseline_clauses)})"

    start = time.time()

    ds_list = xds_from_ms(
        ms_file,
        columns=("DATA", "FLAG", "ANTENNA1", "ANTENNA2"),
        taql_where=taql_where,
        chunks={"row": chunk_size},
    )

    # Compute to actually load data
    for ds in ds_list:
        ant1, ant2, data, flags = dask.compute(
            ds.ANTENNA1.data, ds.ANTENNA2.data, ds.DATA.data, ds.FLAG.data
        )

    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f} seconds")
    return elapsed


def benchmark_daskms_sequential(ms_file, field_id, baselines, chunk_size):
    """daskms reading entire field, filter in memory - SKIPPED (causes OOM)"""
    print("\n2. Testing daskms sequential (no OR query)...")
    print("   SKIPPED: Would cause OOM on systems with limited RAM")
    print("   (Tries to load entire field into memory)")
    return None


def benchmark_casacore_or_query(ms_file, field_id, baselines):
    """casacore table with TaQL OR query"""
    print("\n3. Testing casacore table with TaQL OR query...")

    # Build OR query
    baseline_clauses = [f"(ANTENNA1=={bl[0]} && ANTENNA2=={bl[1]})" for bl in baselines]
    taql_query = f"FIELD_ID=={field_id} && ({' || '.join(baseline_clauses)})"

    start = time.time()

    with table(ms_file) as t:
        t_sub = t.query(taql_query)

        # Read columns
        ant1 = t_sub.getcol("ANTENNA1")
        ant2 = t_sub.getcol("ANTENNA2")
        data = t_sub.getcol("DATA")
        flags = t_sub.getcol("FLAG")

        t_sub.close()

    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Rows read: {len(ant1)}")
    return elapsed


def benchmark_casacore_sequential(ms_file, field_id, baselines):
    """casacore table - read antennas first, then query each baseline"""
    print("\n4. Testing casacore table (ANTENNA SCAN + INDIVIDUAL QUERIES)...")

    start = time.time()

    with table(ms_file) as t:
        # First: Quick scan to identify which baselines exist
        t_field = t.query(f"FIELD_ID=={field_id}")
        ant1 = t_field.getcol("ANTENNA1")
        ant2 = t_field.getcol("ANTENNA2")
        t_field.close()

        # Second: Query each baseline individually
        baseline_data = {}
        for bl in baselines:
            # Check if baseline exists
            mask = (ant1 == bl[0]) & (ant2 == bl[1])
            if np.any(mask):
                # Query this specific baseline
                taql_query = f"FIELD_ID=={field_id} && ANTENNA1=={bl[0]} && ANTENNA2=={bl[1]}"
                t_bl = t.query(taql_query)
                baseline_data[bl] = {
                    "data": t_bl.getcol("DATA"),
                    "flags": t_bl.getcol("FLAG")
                }
                t_bl.close()

    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Baselines found: {len(baseline_data)}")
    return elapsed


def benchmark_casacore_one_by_one(ms_file, field_id, baselines):
    """casacore table reading baselines one-by-one"""
    print("\n5. Testing casacore table one-by-one...")

    start = time.time()

    baseline_data = {}
    with table(ms_file) as t:
        for bl in baselines:
            taql_query = f"FIELD_ID=={field_id} && ANTENNA1=={bl[0]} && ANTENNA2=={bl[1]}"
            t_bl = t.query(taql_query)

            baseline_data[bl] = {
                "data": t_bl.getcol("DATA"),
                "flags": t_bl.getcol("FLAG")
            }
            t_bl.close()

    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f} seconds")
    return elapsed


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_ms_reads.py <ms_file>")
        sys.exit(1)

    ms_file = sys.argv[1]

    print("=" * 70)
    print("MS READING BENCHMARK")
    print("=" * 70)
    print(f"MS file: {ms_file}")

    # Get first field and first 25 baselines for testing
    print("\nDiscovering field and baselines...")
    with table(ms_file) as t:
        field_id = t.getcol("FIELD_ID")[0]
        t_field = t.query(f"FIELD_ID=={field_id}")
        ant1 = t_field.getcol("ANTENNA1")
        ant2 = t_field.getcol("ANTENNA2")
        all_baselines = list(set(zip(ant1, ant2)))
        t_field.close()

    # Test with first 25 baselines
    test_baselines = all_baselines[:25]
    chunk_size = 200000

    print(f"Field ID: {field_id}")
    print(f"Total baselines in field: {len(all_baselines)}")
    print(f"Testing with: {len(test_baselines)} baselines")
    print(f"Chunk size: {chunk_size}")

    # Run benchmarks
    results = {}

    try:
        results["daskms_or"] = benchmark_daskms_or_query(ms_file, field_id, test_baselines, chunk_size)
    except Exception as e:
        print(f"   FAILED: {e}")
        results["daskms_or"] = None

    try:
        results["daskms_seq"] = benchmark_daskms_sequential(ms_file, field_id, test_baselines, chunk_size)
    except Exception as e:
        print(f"   FAILED: {e}")
        results["daskms_seq"] = None

    try:
        results["casacore_or"] = benchmark_casacore_or_query(ms_file, field_id, test_baselines)
    except Exception as e:
        print(f"   FAILED: {e}")
        results["casacore_or"] = None

    try:
        results["casacore_seq"] = benchmark_casacore_sequential(ms_file, field_id, test_baselines)
    except Exception as e:
        print(f"   FAILED: {e}")
        results["casacore_seq"] = None

    try:
        results["casacore_1by1"] = benchmark_casacore_one_by_one(ms_file, field_id, test_baselines)
    except Exception as e:
        print(f"   FAILED: {e}")
        results["casacore_1by1"] = None

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Time (sec)':<15} {'Speedup':<10}")
    print("-" * 70)

    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        baseline_time = results.get("daskms_or") or min(valid_results.values())

        for name, time_val in results.items():
            if time_val is not None:
                speedup = baseline_time / time_val if time_val > 0 else 0
                speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
                print(f"{name:<30} {time_val:<15.2f} {speedup_str:<10}")
            else:
                print(f"{name:<30} {'FAILED':<15} {'-':<10}")

        # Find fastest
        fastest = min(valid_results.items(), key=lambda x: x[1])
        print("\n" + "=" * 70)
        print(f"🏆 FASTEST: {fastest[0]} ({fastest[1]:.2f} seconds)")
        print("=" * 70)
    else:
        print("All benchmarks failed!")

if __name__ == "__main__":
    main()
