# CATBOSS - Radio Astronomy RFI Flagging Suite

**CATBOSS** is a Python-based RFI flagging suite for radio astronomy Measurement Set data.

## POOH: Parallelized Optimized Outlier Hunter

POOH is a multi-scale SumThreshold RFI flagger with bandpass normalization and polynomial fitting.

**Key features:**
- Multi-scale SumThreshold detection with configurable window sizes
- Bandpass normalization via polynomial fitting
- Robust statistics using median and MAD
- RFI channel detection from polynomial residuals
- CPU/GPU hybrid processing with automatic detection
- Numba JIT optimization (10-100× speedup on CPU)
- CUDA acceleration when available

---

## Installation

### Requirements

```
Python >= 3.7
numpy >= 1.20.0
numba
scipy
matplotlib
dask, dask-ms
xarray
tqdm
psutil
```

### Install

```bash
git clone <repository_url>
cd catboss
pip install -e .
```

### GPU Support (Optional)

For GPU acceleration:

```bash
# Verify CUDA
nvidia-smi

# Install CUDA-enabled Numba
conda install numba cudatoolkit

# Test
python -c "from numba import cuda; print(f'GPU: {cuda.is_available()}')"
```

---

## Usage

### Getting Help

```bash
# General help
catboss -h

# POOH-specific help with all options
catboss --cat pooh -h
```

### Command Line

```bash
# Basic usage
catboss --cat pooh data.ms --apply-flags

# Full options
catboss --cat pooh data.ms \
  --combinations 1,2,4,8,16,32,64 \
  --sigma 6.0 \
  --rho 1.5 \
  --poly-degree 5 \
  --deviation-threshold 3.0 \
  --polarizations 0,3 \
  --apply-flags \
  --diagnostic-plots \
  --output-dir rfi_plots \
  --max-threads 32 \
  --max-memory-usage 0.8 \
  --verbose

# Using config file
catboss --cat pooh data.ms --config config.json
```

### Configuration File

Create `config.json`:

```json
{
  "combinations": [1, 2, 4, 8, 16, 32, 64],
  "sigma_factor": 6.0,
  "rho": 1.5,
  "poly_degree": 5,
  "deviation_threshold": 3.0,
  "corr_to_process": [0, 3],
  "apply_flags": true,
  "diagnostic_plots": true,
  "output_dir": "rfi_plots",
  "max_threads": 32,
  "max_memory_usage": 0.8,
  "verbose": true
}
```

Then run:
```bash
catboss --cat pooh data.ms --config config.json
```

### Python API

```python
from catboss.pooh.pooh import hunt_ms, print_gpu_info

# Check hardware
print_gpu_info()

# Configure
options = {
    'combinations': [1, 2, 4, 8, 16, 32, 64],
    'sigma_factor': 6.0,
    'rho': 1.5,
    'poly_degree': 5,
    'deviation_threshold': 3.0,
    'corr_to_process': [0, 3],
    'apply_flags': True,
    'diagnostic_plots': True,
    'output_dir': 'outputs',
    'max_threads': 16,
    'max_memory_usage': 0.8,
    'verbose': True
}

# Run
results = hunt_ms('data.ms', options)

# Results
print(f"Time: {results['total_processing_time']:.2f}s")
print(f"Flagged: {results['overall_percent_flagged']:.2f}%")
print(f"New flags: {results['new_percent_flagged']:.2f}%")
```

---

## Configuration

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `combinations` | list[int] | `[1,2,4,8,16]` | Window sizes for multi-scale detection |
| `sigma_factor` | float | `6.0` | Threshold multiplier |
| `rho` | float | `1.5` | Reduction factor for larger windows |
| `poly_degree` | int | `5` | Polynomial degree for bandpass (3-9) |
| `deviation_threshold` | float | `3.0` | RFI channel detection threshold |
| `corr_to_process` | list[int] | `None` | Correlations to process (None = [0, last]) |
| `apply_flags` | bool | `False` | Write flags to MS |
| `diagnostic_plots` | bool | `False` | Generate diagnostic plots |
| `output_dir` | str | `'outputs'` | Plot output directory |
| `max_threads` | int | `16` | Threads for bandpass processing |
| `max_memory_usage` | float | `0.8` | Fraction of memory to use |
| `verbose` | bool | `False` | Verbose output |

---

## Algorithm

### Processing Pipeline

```
1. Bandpass Normalization
   - Calculate median bandpass per channel
   - Fit polynomial (degree 3-9)
   - Detect RFI channels via residual analysis
   - Normalize amplitudes

2. Threshold Calculation
   - Per-channel statistics (median, MAD)
   - Calculate robust thresholds
   - Scale by window: threshold / (rho^log2(M))

3. Multi-Scale SumThreshold
   For each window size M:
     - Slide window in time direction (skip already flagged points)
     - Slide window in frequency direction (skip already flagged points)
     - Flag outliers exceeding adaptive threshold

4. Flag Combination
   - Simple OR: final_flags = existing_flags | new_flags
   - Existing flags always preserved
   - Write combined flags to MS if requested
```

### Threshold Scaling

For window size M:
```
threshold(M) = base_threshold / (rho ^ log2(M))
```

Example with `rho=1.5`:
- M=1: threshold = base
- M=2: threshold = base / 1.5
- M=4: threshold = base / 2.25
- M=8: threshold = base / 3.375

---

## Output

### Flags

When `apply_flags=True`, flags written to MS FLAG column using logical OR:
```python
MS/FLAG[time, freq, corr] = existing_flags | pooh_flags
```

### Diagnostic Plots

When `diagnostic_plots=True`, generates PNG files:

**Format:** `{output_dir}/pooh_field{N}_bl{A}-{B}_pol{P}.png`

**Content:**
- Left: Raw dynamic spectra
- Right: Flagged spectra (white = flagged)
- Title: Baseline, field, statistics

---

## Performance

### Hardware Comparison

| Mode | Speed | Notes |
|------|-------|-------|
| GPU (CUDA) | Fastest | 10-100× speedup, requires NVIDIA GPU |
| CPU (Numba JIT) | Fast | 10-100× faster than pure Python |
| CPU (Pure Python) | Slow | Not recommended |

### Optimization

**Memory errors:**
```python
options['max_memory_usage'] = 0.6  # Reduce from 0.8
options['combinations'] = [1, 2, 4, 8]  # Fewer windows
options['diagnostic_plots'] = False  # Disable plots
```

**Speed up processing:**
```python
options['combinations'] = [1, 2, 4, 8]  # Fast mode
options['corr_to_process'] = [0]  # Single correlation
options['max_threads'] = 64  # More threads (CPU)
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: catboss` | Run `pip install -e .` |
| `CUDA not available` | CPU mode activates automatically |
| `MemoryError` | Reduce `max_memory_usage` or `combinations` |
| `No baselines found` | Check MS with `taql` |
| GPU not detected | Reinstall: `conda install numba cudatoolkit` |

### Debug

```python
from catboss.pooh.pooh import print_gpu_info
print_gpu_info()

from daskms import xds_from_ms
ds = xds_from_ms('data.ms')[0]
print(f"Shape: {ds.DATA.shape}")
```

---

## Citation

```bibtex
@software{catboss2025,
  title={POOH: Parallelized Optimized Outlier Hunter},
  author={Pal, Arpan},
  year={2025},
  institution={National Centre for Radio Astrophysics, TIFR}
}
```

---

## Development

Developed by Arpan Pal at the National Centre for Radio Astrophysics (NCRA), Tata Institute of Fundamental Research (TIFR).

**Contributing:**
1. Fork repository
2. Create feature branch
3. Submit pull request

**Contact:** For bugs and feature requests, open an issue on GitHub.

---

## License

See LICENSE file.
