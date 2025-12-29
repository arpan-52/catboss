#!/usr/bin/env python3
"""
GPU Diagnostic Script for CATBOSS
This will help identify what's broken in your CUDA/GPU setup
"""

import sys
import numpy as np

print("=" * 70)
print("CATBOSS GPU Diagnostic Test")
print("=" * 70)

# Test 1: Basic CUDA import
print("\n[TEST 1] Importing numba.cuda...")
try:
    from numba import cuda, jit, prange
    print("✓ PASS - numba.cuda imported successfully")
except Exception as e:
    print(f"✗ FAIL - Could not import numba.cuda: {e}")
    sys.exit(1)

# Test 2: Check if GPU is available
print("\n[TEST 2] Checking if GPU is available...")
try:
    available = cuda.is_available()
    if available:
        print("✓ PASS - GPU is available")
    else:
        print("✗ FAIL - No GPU detected")
        sys.exit(1)
except Exception as e:
    print(f"✗ FAIL - Error checking GPU availability: {e}")
    sys.exit(1)

# Test 3: Get CUDA context (FIRST TIME)
print("\n[TEST 3] Creating CUDA context (first time)...")
try:
    ctx = cuda.current_context()
    print("✓ PASS - CUDA context created")
except Exception as e:
    print(f"✗ FAIL - Could not create CUDA context: {e}")
    sys.exit(1)

# Test 4: Get GPU memory info (FIRST TIME)
print("\n[TEST 4] Getting GPU memory info (first time)...")
try:
    free, total = ctx.get_memory_info()
    print(f"✓ PASS - GPU Memory: {total/1e9:.2f} GB total, {free/1e9:.2f} GB free")
except Exception as e:
    print(f"✗ FAIL - Could not get memory info: {e}")
    print("   This is where your GPU is failing!")
    sys.exit(1)

# Test 5: Get memory info AGAIN (SECOND TIME - this is what fails for you!)
print("\n[TEST 5] Getting GPU memory info (SECOND time)...")
print("   THIS IS THE CRITICAL TEST - your segfault happens here!")
try:
    free2, total2 = ctx.get_memory_info()
    print(f"✓ PASS - GPU Memory: {total2/1e9:.2f} GB total, {free2/1e9:.2f} GB free")
except Exception as e:
    print(f"✗ FAIL - Second memory query failed: {e}")
    print("   Your CUDA driver has a bug - context works once but fails on reuse")
    sys.exit(1)

# Test 6: Create CUDA stream
print("\n[TEST 6] Creating CUDA stream...")
try:
    stream = cuda.stream()
    print("✓ PASS - CUDA stream created")
except Exception as e:
    print(f"✗ FAIL - Could not create CUDA stream: {e}")
    print("   This might be the problem!")
    sys.exit(1)

# Test 7: Synchronize CUDA
print("\n[TEST 7] Synchronizing CUDA...")
try:
    cuda.synchronize()
    print("✓ PASS - CUDA synchronized")
except Exception as e:
    print(f"✗ FAIL - Could not synchronize CUDA: {e}")
    sys.exit(1)

# Test 8: Allocate GPU memory
print("\n[TEST 8] Allocating GPU memory...")
try:
    d_array = cuda.to_device(np.ones(1000, dtype=np.float32))
    print("✓ PASS - Allocated 1000 floats on GPU")
except Exception as e:
    print(f"✗ FAIL - Could not allocate GPU memory: {e}")
    sys.exit(1)

# Test 9: Copy data back from GPU
print("\n[TEST 9] Copying data from GPU...")
try:
    h_array = d_array.copy_to_host()
    print(f"✓ PASS - Copied {len(h_array)} elements from GPU")
except Exception as e:
    print(f"✗ FAIL - Could not copy from GPU: {e}")
    sys.exit(1)

# Test 10: Test Numba JIT with parallel
print("\n[TEST 10] Testing Numba JIT with parallel...")
try:
    @jit(nopython=True, parallel=True)
    def test_parallel(arr):
        result = np.zeros_like(arr)
        for i in prange(len(arr)):
            result[i] = arr[i] * 2
        return result

    test_arr = np.arange(1000, dtype=np.float32)
    result = test_parallel(test_arr)
    print(f"✓ PASS - Numba parallel processing works")
except Exception as e:
    print(f"✗ FAIL - Numba parallel failed: {e}")
    sys.exit(1)

# Test 11: ThreadPoolExecutor + CUDA (this might be the issue!)
print("\n[TEST 11] Testing ThreadPoolExecutor with CUDA...")
try:
    from concurrent.futures import ThreadPoolExecutor

    def worker_func(i):
        # Try to use CUDA in a worker thread
        arr = np.ones(100, dtype=np.float32) * i
        return arr.sum()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker_func, range(5)))

    print(f"✓ PASS - ThreadPoolExecutor works: {results}")
except Exception as e:
    print(f"✗ FAIL - ThreadPoolExecutor failed: {e}")
    print("   This might be causing the segfault!")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED! Your GPU setup looks good!")
print("=" * 70)
print("\nIf this script passes but catboss still crashes, the issue is:")
print("1. Complex interaction between ThreadPoolExecutor + Numba parallel + CUDA")
print("2. Or the specific CUDA operations in catboss")
print("\nNext step: Run this and tell me which test fails!")
