#!/usr/bin/env python
"""
Quick test of simple casacore TaQL reading (read all, filter in memory)
Usage: python test_simple_taql.py <ms_file>
"""

import sys
import time
import numpy as np
from casacore.tables import table

if len(sys.argv) < 2:
    print("Usage: python test_simple_taql.py <ms_file>")
    sys.exit(1)

ms_file = sys.argv[1]
field_id = 0

print("=" * 70)
print("SIMPLE TaQL READ TEST")
print("=" * 70)
print(f"MS file: {ms_file}")
print(f"Field ID: {field_id}")
print()

# Discover baselines
print("Discovering baselines...")
with table(ms_file) as t:
    t_field = t.query(f"FIELD_ID=={field_id}")
    ant1 = t_field.getcol("ANTENNA1")
    ant2 = t_field.getcol("ANTENNA2")
    all_baselines = list(set(zip(ant1, ant2)))
    t_field.close()

test_baselines = all_baselines[:25]  # Test with first 25
print(f"Total baselines in field: {len(all_baselines)}")
print(f"Testing with: {len(test_baselines)} baselines")
print()

print("Starting read test...")
print("Method: Read ENTIRE field, filter in memory")
print()

start = time.time()

with table(ms_file) as t:
    # Simple query: just field ID (NO complex OR query)
    t_field = t.query(f"FIELD_ID=={field_id}")

    # Read ALL columns for the field
    print("  Reading ANTENNA1, ANTENNA2...")
    ant1 = t_field.getcol("ANTENNA1")
    ant2 = t_field.getcol("ANTENNA2")

    print("  Reading DATA...")
    data = t_field.getcol("DATA")

    print("  Reading FLAG...")
    flags = t_field.getcol("FLAG")

    # Filter by baseline in memory (FAST!)
    print("  Filtering baselines in memory...")
    baseline_data = {}
    for bl in test_baselines:
        mask = (ant1 == bl[0]) & (ant2 == bl[1])
        if np.any(mask):
            baseline_data[bl] = {
                "data": data[mask],
                "flags": flags[mask]
            }

    t_field.close()

elapsed = time.time() - start

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print(f"Time: {elapsed:.2f} seconds")
print(f"Total rows read: {len(ant1):,}")
print(f"Baselines extracted: {len(baseline_data)}")
print()
print(f"Average per baseline: {elapsed / len(test_baselines):.2f} seconds")
print("=" * 70)
