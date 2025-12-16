# NAMI Architecture

## Design Philosophy

Modular, reusable components with clear separation of concerns.

## Structure

```
nami/
├── nami/
│   ├── __init__.py              # Python API
│   ├── navigator.py             # High-level pipeline (Python)
│   │
│   ├── io/                      # I/O Layer (reusable by other tools)
│   │   ├── __init__.py
│   │   ├── ms_reader.py         # Python interface
│   │   └── cpp/
│   │       ├── ms_reader.cpp    # Fast MS reading
│   │       ├── ms_writer.cpp    # Fast MS writing
│   │       └── ms_iterator.cpp  # Streaming iterator
│   │
│   ├── core/                    # Processing Core
│   │   ├── __init__.py
│   │   ├── fitting.py           # Python interface
│   │   └── cpp/
│   │       ├── uv_calc.cpp      # UV distance calculations
│   │       ├── poly_fit.cpp     # Monotonic polynomial fitting
│   │       ├── outliers.cpp     # Outlier detection
│   │       └── simd_ops.cpp     # SIMD-optimized operations
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── stats.py             # Statistical functions
│
├── setup.py                     # Smart build system
├── find_casacore.py             # Casacore detection
├── pyproject.toml               # Modern Python packaging
└── CMakeLists.txt               # Optional: CMake for complex builds
```

## Module Responsibilities

### `nami.io` - I/O Layer (Reusable)

**Purpose**: Fast MS file access, reusable by ANY radio astronomy tool.

**API**:
```python
from nami.io import MSReader, MSWriter, MSIterator

# Read time chunk
reader = MSReader('observation.ms')
data = reader.read_chunk(time_start, time_end,
                         fields=[0,1], spws=[0])

# Streaming (for huge files)
for chunk in MSIterator('observation.ms', chunk_size_mb=100):
    process(chunk.data, chunk.flags, chunk.uvw)

# Write flags
writer = MSWriter('observation.ms')
writer.write_flags(rows, flags)
```

**Other tools can use**: Any Python code needing fast MS access.

### `nami.core` - Processing Core

**Purpose**: RFI detection algorithms (specific to NAMI).

**API**:
```python
from nami.core import UVCalculator, MonotonicFitter, OutlierDetector

# Calculate UV distances (fast C++)
uv_calc = UVCalculator()
uv_dists = uv_calc.compute(uvw, frequencies)

# Fit model
fitter = MonotonicFitter(degree=3, method='polynomial')
model = fitter.fit(uv_dists, amplitudes, weights)

# Detect outliers
detector = OutlierDetector(sigma=5.0)
flags = detector.flag(amplitudes, model.predict(uv_dists))
```

### `nami.navigator` - High-level Pipeline

**Purpose**: Orchestrate the full RFI flagging pipeline.

Stays in Python for flexibility - calls optimized C++ underneath.

## Benefits

1. **Modular**: Each component is independent
2. **Reusable**: `nami.io` can be used by other tools
3. **Testable**: Each module tested separately
4. **Maintainable**: Clear boundaries
5. **Extensible**: Easy to add new algorithms

## Other Tools Can Use NAMI I/O

Example - someone else's calibration tool:

```python
# They don't need NAMI's flagging, just fast MS I/O
from nami.io import MSReader

reader = MSReader('my_data.ms')
data = reader.read_all_fields()
# ... their custom calibration code ...
```

## Performance

- **I/O layer**: 5-10x faster than python-casacore alone
- **Core algorithms**: 50-200x faster than pure Python
- **Pipeline**: Limited by I/O for small data, by compute for large data
