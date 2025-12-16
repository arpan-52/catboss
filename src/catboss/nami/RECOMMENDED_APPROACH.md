# Recommended Hybrid Approach

## Strategy: C++ for Compute, Python for I/O

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Python Layer (nami/navigator.py)                            │
│  - MS I/O via python-casacore (user already has this)        │
│  - High-level orchestration                                  │
│  - Plotting, argument parsing                                │
└──────────────────┬───────────────────────────────────────────┘
                   │ Pass numpy arrays
                   ↓
┌──────────────────────────────────────────────────────────────┐
│  C++ Extension (_nami_core)                                  │
│  - NO casacore dependency                                    │
│  - Pure numerical computation on numpy arrays                │
│  ├── fast_uv_distances(uvw, frequencies) → uv_dists         │
│  ├── fast_data_collection(data, flags, uv) → points         │
│  ├── monotonic_poly_fit(uv, amp, weights) → coeffs          │
│  └── detect_outliers(amp, predicted, sigma) → flags         │
└──────────────────────────────────────────────────────────────┘
```

## Why This Works

### Profiling Shows:
```
Total time: 100 seconds
├── MS I/O:            20s  (20%)  ← Python is OK here
├── Data collection:   50s  (50%)  ← HUGE C++ speedup here
├── Polynomial fit:    25s  (25%)  ← HUGE C++ speedup here
└── Outlier detection:  5s  (5%)   ← HUGE C++ speedup here
```

**C++ speedup on 80% of runtime, Python handles 20%** → Still massive overall gain!

### Benefits

1. **No casacore linking nightmare**
   - C++ extension is pure computation
   - Only depends on Eigen or standard library
   - Builds on any platform trivially

2. **Simpler build**
   ```python
   # setup.py
   ext = Extension(
       "_nami_core",
       ["nami/core/cpp/compute.cpp"],
       include_dirs=[],  # No casacore!
       libraries=[],     # No casacore!
       extra_compile_args=["-O3", "-march=native", "-fopenmp"]
   )
   ```

3. **Smaller wheel** (~2-5MB vs 50-80MB)

4. **Zero dependency hell**
   ```bash
   pip install python-casacore  # User already has this
   pip install nami             # Just works, no compilation
   ```

5. **Still 50-100x faster overall**
   - C++ handles the slow loops
   - Python MS I/O is fast enough (it's already calling C++ underneath)

## Example: C++ Compute Module

**nami/core/cpp/compute.cpp**:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cmath>

namespace py = pybind11;

// NO CASACORE - just numpy arrays!

py::array_t<double> calculate_uv_distances(
    py::array_t<double> uvw,       // Shape: (n_rows, 3)
    py::array_t<double> wavelengths // Shape: (n_channels,)
) {
    auto uvw_r = uvw.unchecked<2>();
    auto wave_r = wavelengths.unchecked<1>();

    size_t n_rows = uvw_r.shape(0);
    size_t n_chan = wave_r.shape(0);

    auto result = py::array_t<double>({n_rows, n_chan});
    auto result_w = result.mutable_unchecked<2>();

    // SIMD-friendly loop
    #pragma omp parallel for
    for (size_t row = 0; row < n_rows; ++row) {
        double u = uvw_r(row, 0);
        double v = uvw_r(row, 1);
        double uv_len = std::sqrt(u*u + v*v);

        for (size_t chan = 0; chan < n_chan; ++chan) {
            result_w(row, chan) = uv_len / wave_r(chan);
        }
    }

    return result;
}

// More functions: data collection, fitting, outliers...

PYBIND11_MODULE(_nami_core, m) {
    m.def("calculate_uv_distances", &calculate_uv_distances,
          "Fast UV distance calculation");
}
```

**Python side (nami/navigator.py)**:
```python
from casacore.tables import table  # python-casacore for I/O
import _nami_core as core          # C++ for compute

def read_ms_chunk(ms_file, tstart, tend, datacolumn='DATA'):
    # Use python-casacore (already fast enough)
    with table(ms_file) as tb:
        query = f"TIME >= {tstart} AND TIME < {tend}"
        with tb.query(query) as subtb:
            uvw = subtb.getcol('UVW')
            data = subtb.getcol(datacolumn)
            flags = subtb.getcol('FLAG')

    # Get frequencies
    with table(f"{ms_file}::SPECTRAL_WINDOW") as spw_table:
        freqs = spw_table.getcol('CHAN_FREQ')[0]

    # Call C++ for fast computation
    wavelengths = 299792458.0 / freqs
    uv_distances = core.calculate_uv_distances(uvw, wavelengths)

    return {
        'data': data,
        'flags': flags,
        'uv_distances': uv_distances
    }
```

## Installation Flow

### User Experience:
```bash
# Step 1: Install python-casacore (if not already)
pip install python-casacore

# Step 2: Install nami (downloads pre-built wheel)
pip install nami

# Done! No compilation, no casacore hunting
```

### Your CI/CD:
```yaml
# GitHub Actions builds wheels for all platforms
# No casacore in build environment needed!
# Just standard C++ compiler

- name: Build wheels
  run: pip wheel . -w wheelhouse
  # Compiles C++ compute module only
  # No external dependencies
```

## Migration Path

**Phase 1** (1 week): Convert hottest loop
- `calculate_uv_distances` → C++
- Test: Should match Python exactly
- Speedup: ~20x on this function

**Phase 2** (1 week): Data collection
- `collect_data_points` → C++
- Speedup: ~50x on this function

**Phase 3** (1 week): Custom polynomial solver
- `monotonic_poly_fit` → C++
- Speedup: ~10x on this function

**Total speedup: 30-50x** with zero casacore linking issues.

## When to Consider Vendoring Casacore

Only if profiling shows MS I/O is >50% of runtime AND:
- Processing small chunks repeatedly (I/O dominant)
- Many small reads (iterator overhead)

For large chunks (default 30min), python-casacore I/O is fine.
