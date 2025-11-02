# Installation Guide

## Package Information

- **Package Name**: `llm-instrumentation`
- **Version**: 0.1.0
- **Python Version**: ≥3.8

## Installation Methods

### 1. Install from Local Build

If you have the built distribution files:

```bash
pip install dist/llm_instrumentation-0.1.0-py3-none-any.whl
```

### 2. Install in Development Mode

For development and testing:

```bash
cd llm-instrumentation
pip install -e .
```

### 3. Install with Optional Dependencies

The package has optional dependency groups:

```bash
# Install with development tools
pip install llm-instrumentation[dev]

# Install with eBPF monitoring support
pip install llm-instrumentation[ebpf]

# Install everything
pip install llm-instrumentation[all]
```

## Building from Source

To build the package yourself:

### Prerequisites

```bash
pip install build wheel setuptools
```

### Build Commands

```bash
cd llm-instrumentation

# Build both wheel and source distribution
python -m build

# Build only wheel
python -m build --wheel

# Build only source distribution
python -m build --sdist
```

The built packages will be in the `dist/` directory:
- `llm_instrumentation-0.1.0-py3-none-any.whl` - Wheel distribution
- `llm_instrumentation-0.1.0.tar.gz` - Source distribution

## Verifying Installation

After installation, verify it works:

```python
import llm_instrumentation
print(f"Version: {llm_instrumentation.__version__}")

# Check available exports
from llm_instrumentation import (
    InstrumentationFramework,
    InstrumentationConfig,
    HookGranularity,
    HookConfig,
    StreamingSerializer,
    TensorCompressionManager,
)
```

## Dependencies

### Core Dependencies

- `torch>=2.0.0` - PyTorch for model instrumentation
- `transformers>=4.30.0` - Hugging Face Transformers
- `numpy>=1.24.0` - Numerical operations
- `lz4>=4.0.0` - LZ4 compression
- `zstandard>=0.20.0` - Zstandard compression
- `aiofiles>=23.0.0` - Async file operations
- `psutil>=5.9.0` - System monitoring
- `networkx>=3.0` - Graph analysis
- `matplotlib>=3.5.0` - Visualization
- `jupyter>=1.0.0` - Notebook support
- `ipython>=8.0.0` - Interactive Python

### Optional Dependencies

**Development** (`[dev]`):
- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `black>=23.0.0` - Code formatter
- `flake8>=6.0.0` - Linter
- `mypy>=1.0.0` - Type checker

**eBPF Monitoring** (`[ebpf]`):
- `bcc>=0.1.0` - BPF Compiler Collection (requires root privileges)

## Publishing to PyPI

To publish the package to PyPI:

```bash
# Install twine
pip install twine

# Upload to Test PyPI (recommended first)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Permission Issues with eBPF

The eBPF monitoring scripts (`tracepoints.py`, `system_metrics.py`) require root privileges:

```bash
sudo python scripts/tracepoints.py
```

### PyTorch Installation

PyTorch has different versions for different platforms. See the [official installation guide](https://pytorch.org/get-started/locally/) for platform-specific instructions.

## Next Steps

After installation, see:
- [README.md](README.md) - Package overview and quick start
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - Detailed usage guide
- [docs/API.md](docs/API.md) - API reference
- [examples/](examples/) - Example scripts
