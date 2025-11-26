# CATBOSS - Radio Astronomy RFI Flagging Suite

**CATBOSS** (Comprehensive Astronomical Tools for Baseband Observation Signal Scanning) is a Python-based RFI (Radio Frequency Interference) flagging suite designed for radio astronomy data stored in Measurement Set (MS) format.

The flagship algorithm is **POOH** (Parallelized Optimized Outlier Hunter), a sophisticated SumThreshold-based RFI flagger with bandpass normalization and polynomial fitting capabilities.

> _"Pooh is my first cat. He specializes in collecting snakes and birds, which he thoughtfully deposits on my bed. I write this code in return of those gifts."_
>  Arpan Pal, NCRA-TIFR, 2025

---

## Features

- **Dual Algorithm Support**:
  - `pooh`: Full POOH implementation with bandpass normalization and polynomial fitting
  - `bp`: Simplified POOH variant without bandpass normalization (faster processing)

- **CPU/GPU Hybrid Processing**:
  - Automatic hardware detection
  - GPU acceleration via CUDA (Numba)
  - Graceful CPU fallback with Numba JIT optimization (10-100x speedup over pure Python)

- **Advanced Flagging**:
  - Multi-scale SumThreshold algorithm with configurable window sizes
  - Channel-specific threshold calculation
  - Robust outlier detection using median and MAD statistics
  - Independent or combined correlation processing
  - Preserves existing flags while adding new ones

- **Memory Optimization**:
  - Intelligent batch processing based on available GPU/system memory
  - Baseline chunking for large datasets
  - Parallel processing with CUDA streams

- **Diagnostic Outputs**:
  - Optional diagnostic plots showing raw and flagged dynamic spectra
  - Comprehensive flagging statistics
  - Per-baseline and per-correlation analysis

---

## Installation

### Requirements

- Python e 3.7
- NumPy e 1.20.0
- Numba (for CPU JIT and optional GPU acceleration)
- Matplotlib (for diagnostic plots)
- Dask and dask-ms (for MS file I/O)
- XArray
- tqdm (progress bars)
- psutil (memory monitoring)
- SciPy (for polynomial fitting in POOH variant)

### Install from source

```bash
git clone <repository_url>
cd catboss
pip install -e .
```

This will install catboss and all required dependencies.

### GPU Support (Optional)

For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA Toolkit (10.0 or later)
- Numba with CUDA support (`conda install numba cudatoolkit`)

CATBOSS will automatically detect GPU availability and use it if present.

---

## Usage

CATBOSS provides two main algorithms that can be imported and used in Python scripts or notebooks:

### Basic Usage Example

```python
from catboss.pooh.pooh import hunt_ms, print_gpu_info

# Print hardware information
print_gpu_info()

# Configure processing options
options = {
    'combinations': [1, 2, 4, 8, 16, 32, 64],  # Window sizes
    'sigma_factor': 6.0,                        # Threshold multiplier
    'rho': 1.5,                                 # Scale factor for larger windows
    'corr_to_process': [0, 3],                  # Correlations to process (None = default)
    'apply_flags': True,                        # Write flags back to MS
    'diagnostic_plots': True,                   # Generate diagnostic plots
    'output_dir': 'outputs',                    # Directory for diagnostic plots
    'poly_degree': 5,                           # Polynomial degree for bandpass fitting (POOH only)
    'deviation_threshold': 3.0,                 # RFI detection threshold (POOH only)
    'max_memory_usage': 0.8,                    # Maximum fraction of memory to use
    'max_threads': 16,                          # Max threads for parallel bandpass processing
    'verbose': False                            # Verbose output
}

# Run POOH flagger
results = hunt_ms('path/to/your_data.ms', options)

# Print results
print(f"\nFlagging Summary:")
print(f"Total time: {results['total_processing_time']:.2f} seconds")
print(f"Overall flagging: {results['overall_percent_flagged']:.2f}%")
print(f"New flags added: {results['new_percent_flagged']:.2f}%")
print(f"Baselines processed: {results['baselines_processed']}")
print(f"Baselines skipped: {results['baselines_skipped']}")
```

### Using the BP Variant (Without Bandpass Normalization)

```python
from catboss.pooh.bp import hunt_ms, print_gpu_info

# BP variant uses the same interface but without bandpass-specific options
options = {
    'combinations': [1, 2, 4, 8, 16],
    'sigma_factor': 6.0,
    'rho': 1.5,
    'corr_to_process': None,          # None = default (process correlations 0 and last)
    'apply_flags': True,
    'diagnostic_plots': False,
    'output_dir': 'outputs',
    'max_memory_usage': 0.8,
    'verbose': True
}

results = hunt_ms('path/to/your_data.ms', options)
```

---

## Configuration Options

### Required Options
- **combinations**: List of window sizes for SumThreshold (e.g., `[1, 2, 4, 8, 16, 32, 64]`)
  - Smaller windows: detect narrow RFI
  - Larger windows: detect broad RFI

### Threshold Options
- **sigma_factor**: Multiplier for threshold calculation (default: `6.0`)
  - Higher = more conservative (fewer flags)
  - Lower = more aggressive (more flags)

- **rho**: Reduction factor for larger window sizes (default: `1.5`)
  - Thresholds are divided by `rho^(log2(M))` for window size M

### POOH-Specific Options (pooh.py only)
- **poly_degree**: Degree of polynomial for bandpass fitting (default: `5`)
- **deviation_threshold**: Threshold in sigma for RFI channel detection (default: `3.0`)

### Correlation Options
- **corr_to_process**: List of correlation indices to process
  - `None`: Process correlations 0 and last (default, approx. Stokes I)
  - `[0]`: Process only correlation 0
  - `[0, 3]`: Process correlations 0 and 3
  - `[0, 1, 2, 3]`: Process all correlations separately

### Output Options
- **apply_flags**: Whether to write flags back to MS (default: `False`)
- **diagnostic_plots**: Generate diagnostic plots (default: `False`)
- **output_dir**: Directory for diagnostic plots (default: `'outputs'`)

### Performance Options
- **max_memory_usage**: Fraction of available memory to use (default: `0.8`)
- **max_threads**: Maximum threads for parallel processing (POOH only, default: `16`)
- **verbose**: Enable verbose output (default: `False`)

---

## Output Files

### Flags
When `apply_flags=True`, flags are written directly to the FLAG column of the input MS file:
- Existing flags are preserved (logical OR operation)
- New flags are added for detected RFI
- Flags can be applied to specific correlations or propagated across all

### Diagnostic Plots
When `diagnostic_plots=True`, PNG files are generated in the output directory:

**Filename format**: `pooh_field{field_id}_bl{ant1}-{ant2}_pol{corr}.png`

Each plot contains two panels:
- **Left**: Raw dynamic spectra (time × frequency)
- **Right**: Flagged spectra with flags shown in white

Example: `pooh_field0_bl0-1_pol0.png`

---

## Hardware Detection

CATBOSS automatically detects available hardware:

### GPU Mode
```
GPU detected - will use GPU acceleration
=== Hardware Information ===
Number of available GPUs: 1
GPU 0: Tesla V100-SXM2-16GB
  - Compute capability: 7.0
============================
```

### CPU Mode
```
No GPU detected - will use CPU processing
=== Hardware Information ===
Running in CPU mode - no GPU available
============================
```

**Note**: CPU mode uses Numba JIT compilation for 10-100x speedup compared to pure Python.

---

## Algorithm Details

### SumThreshold Algorithm

The core flagging algorithm works as follows:

1. **Statistics Calculation**: Calculate robust per-channel thresholds using median and standard deviation
2. **Bandpass Normalization** (POOH only): Fit polynomial to bandpass, detect and flag RFI channels
3. **Multi-Scale Detection**: For each window size M in combinations:
   - **Time direction**: Sum amplitudes in sliding windows along time axis
   - **Frequency direction**: Sum amplitudes in sliding windows along frequency axis
   - Flag samples exceeding adaptive thresholds
4. **Threshold Scaling**: Larger windows use reduced thresholds: `threshold / (rho^log2(M))`

### Flag Preservation

CATBOSS preserves existing flags in the MS:
- Never unflags previously flagged data
- Uses logical OR to combine existing and new flags
- Handles correlations independently or jointly based on configuration

---

## Example Workflows

### Quick RFI Flagging (CPU Mode)
```python
from catboss.pooh.bp import hunt_ms

# Minimal configuration for fast flagging
options = {
    'combinations': [1, 2, 4, 8],    # Fewer windows = faster
    'sigma_factor': 6.0,
    'rho': 1.5,
    'apply_flags': True,
    'diagnostic_plots': False,
    'verbose': True
}

results = hunt_ms('data.ms', options)
```

### Aggressive RFI Flagging with Diagnostics (GPU Mode)
```python
from catboss.pooh.pooh import hunt_ms, print_gpu_info

print_gpu_info()

# More aggressive flagging with diagnostics
options = {
    'combinations': [1, 2, 4, 8, 16, 32, 64],  # More scales
    'sigma_factor': 4.0,                         # Lower threshold (more aggressive)
    'rho': 1.5,
    'corr_to_process': [0, 3],                   # Process specific correlations
    'apply_flags': True,
    'diagnostic_plots': True,
    'output_dir': 'rfi_plots',
    'poly_degree': 7,                            # Higher degree polynomial
    'deviation_threshold': 2.5,                  # More sensitive RFI detection
    'max_threads': 32,
    'verbose': True
}

results = hunt_ms('data.ms', options)
```

### Processing Specific Baselines
CATBOSS automatically processes all baselines in the MS. To process specific baselines or fields, filter your MS using CASA or TaQL before running CATBOSS.

---

## Performance Tips

1. **GPU vs CPU**:
   - GPU: Optimal for large datasets with many baselines
   - CPU: Numba JIT provides excellent performance for small to medium datasets

2. **Memory Management**:
   - CATBOSS automatically calculates batch sizes based on available memory
   - Reduce `max_memory_usage` if you experience out-of-memory errors
   - For very large datasets, chunking is applied automatically

3. **Window Sizes**:
   - Start with default `[1, 2, 4, 8, 16, 32, 64]`
   - Reduce for faster processing: `[1, 2, 4, 8]`
   - Increase for more thorough detection: `[1, 2, 4, 8, 16, 32, 64, 128]`

4. **Bandpass Normalization** (POOH):
   - Use full POOH variant for data with strong bandpass structure
   - Use BP variant for pre-calibrated data or faster processing

---

## Troubleshooting

### GPU Not Detected
If you have a GPU but CATBOSS uses CPU mode:
```bash
# Check CUDA installation
python -c "from numba import cuda; print(cuda.is_available())"

# If False, reinstall Numba with CUDA support
conda install numba cudatoolkit
```

### Out of Memory Errors
- Reduce `max_memory_usage` (e.g., from 0.8 to 0.6)
- Process fewer window sizes
- Disable diagnostic plots for large datasets

### Slow Performance
- Ensure Numba is installed correctly (check for JIT compilation messages)
- For CPU mode, ensure you're not running other memory-intensive processes
- Consider using fewer combinations or processing fewer correlations

---

## Citation

If you use CATBOSS/POOH in your research, please acknowledge:

```
POOH (Parallelized Optimized Outlier Hunter) RFI Flagger
Developed by Arpan Pal, NCRA-TIFR, 2025
```

---

## License

See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## Contact

For questions, issues, or feature requests, please contact:
- Arpan Pal - NCRA-TIFR

---

## Acknowledgments

CATBOSS is developed at the National Centre for Radio Astrophysics (NCRA), Tata Institute of Fundamental Research (TIFR).

Special thanks to Pooh the cat for continuous motivation and occasional code reviews (mostly in the form of walking across the keyboard).
