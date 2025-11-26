# 🐱 CATBOSS - Radio Astronomy RFI Flagging Suite

**CATBOSS** (Comprehensive Astronomical Tools for Baseband Observation Signal Scanning) is a high-performance Python-based RFI (Radio Frequency Interference) flagging suite for radio astronomy data in Measurement Set (MS) format.

## 🎯 POOH: Parallelized Optimized Outlier Hunter

The flagship algorithm **POOH** is a sophisticated multi-scale SumThreshold-based RFI flagger with:
- **Bandpass normalization** using polynomial fitting
- **Robust outlier detection** with median and MAD statistics
- **Multi-scale analysis** with configurable window sizes
- **Intelligent RFI channel detection** via polynomial residuals

> _"Pooh is my first cat. He specializes in collecting snakes and birds, which he thoughtfully deposits on my bed. I write this code in return of those gifts."_
> — Arpan Pal, NCRA-TIFR, 2025

---

## ✨ Key Features

### 🚀 Hybrid CPU/GPU Processing
- **Automatic hardware detection** - runs on any system
- **GPU acceleration via CUDA** when available (Numba)
- **Numba JIT optimization for CPU** - 10-100× faster than pure Python
- **Graceful fallback** - no crashes on CPU-only systems

### 🎯 Advanced Flagging Capabilities
- **Multi-scale SumThreshold** with configurable window sizes (1 to 128 samples)
- **Bandpass normalization** via polynomial fitting (degree 3-9)
- **Per-channel threshold calculation** using robust statistics
- **RFI channel detection** based on polynomial residuals
- **Independent or combined correlation processing** (Stokes I approximation)
- **Flag preservation** - never unflags existing flags

### 💾 Smart Memory Management
- **Automatic batch sizing** based on GPU/CPU memory
- **Baseline chunking** for datasets larger than memory
- **Parallel processing** with CUDA streams (GPU) or threads (CPU)
- **Efficient I/O** via dask-ms for lazy loading

### 📊 Diagnostic Outputs
- **Dynamic spectra plots** showing raw and flagged data
- **Comprehensive statistics** per baseline and correlation
- **Flagging percentages** and processing times
- **Bandpass visualization** with polynomial fits

---

## 📦 Installation

### Requirements

```bash
Python >= 3.7
numpy >= 1.20.0
numba               # CPU JIT + optional GPU acceleration
matplotlib          # Diagnostic plots
dask, dask-ms       # MS file I/O
xarray              # Data structures
scipy               # Polynomial fitting
tqdm                # Progress bars
psutil              # Memory monitoring
```

### Install from Source

```bash
git clone <repository_url>
cd catboss
pip install -e .
```

This installs CATBOSS and all dependencies.

### GPU Support (Optional but Recommended)

For **10-100× speedup** on large datasets:

```bash
# Check if you have NVIDIA GPU with CUDA
nvidia-smi

# Install CUDA-enabled Numba
conda install numba cudatoolkit

# Verify GPU detection
python -c "from numba import cuda; print(f'GPU Available: {cuda.is_available()}')"
```

✅ **GPU mode**: Processes entire datasets in seconds
⚡ **CPU mode**: Numba JIT provides excellent performance for small/medium datasets

---

## 🎮 Quick Start

### Basic Usage

```python
from catboss.pooh.pooh import hunt_ms, print_gpu_info

# Check hardware (GPU or CPU mode)
print_gpu_info()

# Minimal configuration - works out of the box
options = {
    'combinations': [1, 2, 4, 8, 16],     # Window sizes
    'sigma_factor': 6.0,                  # Threshold multiplier (lower = more aggressive)
    'rho': 1.5,                           # Scale factor for larger windows
    'apply_flags': True,                  # Write flags to MS
    'diagnostic_plots': False,            # Disable plots for speed
    'verbose': True
}

# Run POOH
results = hunt_ms('path/to/data.ms', options)

# Results summary
print(f"\n{'='*50}")
print(f"POOH Flagging Complete!")
print(f"{'='*50}")
print(f"⏱️  Processing time: {results['total_processing_time']:.2f}s")
print(f"🎯 Total flagged: {results['overall_percent_flagged']:.2f}%")
print(f"🆕 New flags: {results['new_percent_flagged']:.2f}%")
print(f"📊 Baselines processed: {results['baselines_processed']}")
print(f"{'='*50}")
```

### Advanced Usage with Diagnostics

```python
from catboss.pooh.pooh import hunt_ms, print_gpu_info

print_gpu_info()

# Full-featured configuration
options = {
    # Multi-scale analysis
    'combinations': [1, 2, 4, 8, 16, 32, 64],

    # Threshold tuning
    'sigma_factor': 5.0,              # More aggressive (4.0-8.0 range)
    'rho': 1.5,                       # Standard scaling

    # Bandpass fitting
    'poly_degree': 5,                 # Polynomial degree (3-9)
    'deviation_threshold': 3.0,       # RFI detection sensitivity (2.0-5.0)

    # Correlation processing
    'corr_to_process': [0, 3],        # Process correlations 0 and 3

    # Output options
    'apply_flags': True,              # Write to MS
    'diagnostic_plots': True,         # Generate plots
    'output_dir': 'rfi_diagnostics',  # Plot directory

    # Performance tuning
    'max_threads': 32,                # Parallel bandpass processing
    'max_memory_usage': 0.8,          # Use 80% of available memory

    # Verbosity
    'verbose': True
}

results = hunt_ms('data.ms', options)

# Diagnostic plots saved to: rfi_diagnostics/pooh_field{N}_bl{A}-{B}_pol{P}.png
```

---

## ⚙️ Configuration Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `combinations` | list[int] | `[1,2,4,8,16]` | Window sizes for multi-scale detection |
| `sigma_factor` | float | `6.0` | Threshold multiplier (lower = more aggressive) |
| `rho` | float | `1.5` | Reduction factor for larger windows |

### Bandpass Normalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poly_degree` | int | `5` | Polynomial degree for bandpass fitting (3-9) |
| `deviation_threshold` | float | `3.0` | RFI channel detection threshold in sigma |

### Correlation Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corr_to_process` | list[int] or None | `None` | Correlations to process |

**Options:**
- `None` → Process correlations 0 and last (Stokes I approximation)
- `[0]` → Process only correlation 0
- `[0, 3]` → Process correlations 0 and 3
- `[0, 1, 2, 3]` → Process all correlations independently

### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_flags` | bool | `False` | Write flags back to MS file |
| `diagnostic_plots` | bool | `False` | Generate diagnostic plots |
| `output_dir` | str | `'outputs'` | Directory for diagnostic plots |

### Performance Tuning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_threads` | int | `16` | Max threads for parallel bandpass processing |
| `max_memory_usage` | float | `0.8` | Fraction of memory to use (0.0-1.0) |
| `verbose` | bool | `False` | Enable detailed output |

---

## 🎨 Understanding the Algorithm

### How POOH Works

```
1. BANDPASS NORMALIZATION
   ├─ Calculate median bandpass per channel
   ├─ Fit polynomial to bandpass (degree 3-9)
   ├─ Detect RFI channels via residual analysis
   └─ Normalize amplitudes by smooth bandpass

2. THRESHOLD CALCULATION
   ├─ Compute per-channel statistics (median, MAD)
   ├─ Calculate robust thresholds
   └─ Scale thresholds by window size: threshold / (rho^log2(M))

3. MULTI-SCALE SUMTHRESHOLD
   For each window size M in [1, 2, 4, 8, 16, ...]:
   ├─ TIME DIRECTION
   │  ├─ Slide window along time axis
   │  ├─ Sum unflagged amplitudes in window
   │  ├─ Compare to adaptive threshold
   │  └─ Flag outliers exceeding threshold
   └─ FREQUENCY DIRECTION
      ├─ Slide window along frequency axis
      ├─ Sum unflagged amplitudes in window
      ├─ Compare to adaptive threshold
      └─ Flag outliers exceeding threshold

4. FLAG COMBINATION
   ├─ Combine flags from all scales (logical OR)
   ├─ Preserve existing flags
   └─ Write to MS file (if apply_flags=True)
```

### Window Size Strategy

| Window Size | Detects | Use Case |
|-------------|---------|----------|
| **1** | Point sources, narrow RFI | Lightning, satellites |
| **2-4** | Short bursts | Aircraft radar |
| **8-16** | Moderate RFI | Digital TV, mobile phones |
| **32-64** | Broadband RFI | FM radio, continuous interference |
| **128+** | Very broad RFI | Entire subbands |

**Recommended combinations:**
- Fast: `[1, 2, 4, 8]`
- Standard: `[1, 2, 4, 8, 16, 32, 64]`
- Thorough: `[1, 2, 4, 8, 16, 32, 64, 128]`

---

## 📈 Output Files

### Flags in MS File

When `apply_flags=True`, flags are written to the `FLAG` column:

```python
# Flags are combined with existing flags (logical OR)
new_flags = existing_flags | pooh_flags

# Per-correlation flagging
MS/FLAG[time, freq, corr] = True  # Flagged
                            = False # Unflagged
```

### Diagnostic Plots

When `diagnostic_plots=True`, PNG files are generated:

**Filename format:**
```
{output_dir}/pooh_field{field}_bl{ant1}-{ant2}_pol{corr}.png
```

**Example:** `rfi_diagnostics/pooh_field0_bl0-15_pol0.png`

**Plot contents:**
- **Left panel**: Raw dynamic spectra (time × frequency)
- **Right panel**: Flagged spectra (flags shown in white)
- **Title**: Baseline info, field ID, flagging statistics

---

## 🖥️ Hardware Detection

POOH automatically detects and uses available hardware:

### GPU Mode (NVIDIA CUDA)

```
GPU detected - will use GPU acceleration
=== Hardware Information ===
Number of available GPUs: 1
GPU 0: Tesla V100-SXM2-32GB
  - Compute capability: 7.0
============================

Processing 450 baselines × 2048 channels...
⚡ Complete in 12.3 seconds
```

### CPU Mode (Numba JIT)

```
No GPU detected - will use CPU processing
=== Hardware Information ===
Running in CPU mode - no GPU available
============================

Processing 450 baselines × 2048 channels...
⚡ Complete in 142.7 seconds (with Numba JIT)
```

**Performance comparison:**
- GPU (CUDA): **Fastest** - 10-100× speedup
- CPU (Numba JIT): **Fast** - 10-100× faster than pure Python
- CPU (Pure Python): **Slow** - not recommended for large datasets

---

## 📚 Example Workflows

### 1. Quick RFI Removal (Conservative)

```python
from catboss.pooh.pooh import hunt_ms

options = {
    'combinations': [1, 2, 4, 8, 16],
    'sigma_factor': 6.0,        # Conservative
    'apply_flags': True,
    'verbose': True
}

results = hunt_ms('observation.ms', options)
# Typically flags 5-15% of data
```

### 2. Aggressive Flagging (Strong RFI Environment)

```python
options = {
    'combinations': [1, 2, 4, 8, 16, 32, 64],
    'sigma_factor': 4.0,        # More aggressive
    'rho': 1.3,                 # Tighter scaling
    'poly_degree': 7,           # Higher polynomial fit
    'deviation_threshold': 2.5, # More sensitive
    'apply_flags': True,
    'diagnostic_plots': True,
    'output_dir': 'aggressive_flags'
}

results = hunt_ms('rfi_heavy.ms', options)
# Typically flags 20-40% of data
```

### 3. Research Mode (Full Diagnostics)

```python
options = {
    'combinations': [1, 2, 4, 8, 16, 32, 64, 128],
    'sigma_factor': 5.0,
    'corr_to_process': [0, 1, 2, 3],  # All correlations
    'apply_flags': False,              # Don't modify MS yet
    'diagnostic_plots': True,          # Full diagnostics
    'output_dir': 'research_plots',
    'verbose': True
}

results = hunt_ms('test_data.ms', options)
# Review plots before applying flags
```

### 4. Production Pipeline (Optimal Performance)

```python
from catboss.pooh.pooh import hunt_ms, print_gpu_info
import sys

# Check hardware
print_gpu_info()

# Optimized settings
options = {
    'combinations': [1, 2, 4, 8, 16, 32],
    'sigma_factor': 5.5,
    'rho': 1.5,
    'poly_degree': 5,
    'deviation_threshold': 3.0,
    'apply_flags': True,
    'diagnostic_plots': False,    # Disable for speed
    'max_threads': 32,
    'max_memory_usage': 0.85,
    'verbose': False
}

try:
    results = hunt_ms(sys.argv[1], options)
    print(f"✓ Flagged {results['new_percent_flagged']:.2f}% in {results['total_processing_time']:.1f}s")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
```

---

## 🔧 Performance Optimization

### Memory Issues

**Problem:** Out of memory errors

**Solutions:**
```python
# Reduce memory usage
options['max_memory_usage'] = 0.6  # Use 60% instead of 80%

# Process fewer scales
options['combinations'] = [1, 2, 4, 8]  # Reduce from [1..64]

# Disable diagnostics
options['diagnostic_plots'] = False
```

### Speed Optimization

**Problem:** Processing too slow

**Solutions:**
```python
# Use fewer window sizes
options['combinations'] = [1, 2, 4, 8]  # Fast mode

# Process fewer correlations
options['corr_to_process'] = [0]  # Single correlation

# Increase threads (CPU mode)
options['max_threads'] = 64

# Check GPU availability
from catboss.pooh.pooh import print_gpu_info
print_gpu_info()
```

### GPU Not Detected

**Problem:** GPU available but not used

**Check CUDA installation:**
```bash
# Verify GPU
nvidia-smi

# Test Numba CUDA
python -c "from numba import cuda; print(cuda.is_available())"

# Reinstall if needed
conda install numba cudatoolkit=11.8
```

---

## 🐛 Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'catboss'` | Not installed | `pip install -e .` |
| `CUDA not available` | No GPU or CUDA | Use CPU mode (automatic) |
| `MemoryError` | Dataset too large | Reduce `max_memory_usage` |
| `ValueError: No baselines found` | Empty MS or wrong field | Check MS with `taql` |
| Plots not generated | `diagnostic_plots=False` | Set to `True` |

### Getting Help

```python
# Print configuration
from catboss.pooh.pooh import print_gpu_info
print_gpu_info()

# Check MS structure
from daskms import xds_from_ms
ds = xds_from_ms('data.ms')[0]
print(f"Shape: {ds.DATA.shape}")  # (rows, channels, correlations)
print(f"Columns: {ds.data_vars}")
```

---

## 📖 Citation

If you use CATBOSS/POOH in your research, please acknowledge:

```bibtex
@software{catboss2025,
  title={POOH: Parallelized Optimized Outlier Hunter for Radio Astronomy RFI Flagging},
  author={Pal, Arpan},
  year={2025},
  institution={National Centre for Radio Astrophysics, TIFR}
}
```

---

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

See LICENSE file for details.

---

## 🙏 Acknowledgments

**CATBOSS** is developed at the **National Centre for Radio Astrophysics (NCRA)**, Tata Institute of Fundamental Research (TIFR).

**Special thanks to Pooh the cat** 🐱 for continuous motivation, code reviews (via keyboard walking), and the occasional snake delivery.

---

## 📬 Contact

**Arpan Pal**
National Centre for Radio Astrophysics (NCRA-TIFR)

For bug reports and feature requests, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ and 🐱 at NCRA-TIFR**

*"In the hunt for RFI, be like Pooh: patient, persistent, and always ready to pounce."*

</div>
