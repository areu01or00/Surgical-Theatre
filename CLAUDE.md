# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SurgicalTheater is a Python package that provides memory-efficient model validation during training. The core innovation is a delta-based context manager that enables frequent validation with reduced GPU memory usage - solving the problem where traditional approaches require 2x model memory (e.g., 40GB for a 20GB model) while SurgicalTheater needs only ~1 parameter set overhead (~2-3GB).

## Key Use Cases

1. **LoRA/PEFT Training**: Validate every 10 steps instead of once per epoch without memory overhead
2. **Reinforcement Learning**: Prevent reward hacking through frequent validation on held-out environments
3. **Budget Hardware**: Train larger models on smaller GPUs (24GB instead of 48GB requirement)

## Core Architecture

### Main Components

- **`SurgicalTheater` class** (`surgical_theater/core.py`): The primary context manager that handles weight backup/restore
- **Convenience function** `surgical_theater()`: Simplified interface for common use cases
- **Examples**: Practical demonstrations in `examples/` directory

### Core Algorithm

The SurgicalTheater works by:
1. **Compute Delta**: Calculate modification deltas for target parameters (~1 parameter set)
2. **Apply**: Apply deltas in-place to parameters  
3. **Restore**: Automatically subtract deltas to restore original state on exit

Key insight: Instead of copying entire model weights, only store the deltas needed for restoration. This uses ~1 parameter set of extra memory instead of 2x full model.

### Supported Modification Types

- `"scale"`: Apply scaling factors to weights
- `"noise"`: Add noise for robustness testing
- `"disable"`: Zero out layers for ablation studies
- `"custom"`: User-defined modification functions

## Development Commands

### Installation
```bash
# Local development install
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With examples dependencies
pip install -e ".[examples]"
```

### Code Quality
```bash
# Format code
black surgical_theater/ examples/ tests/ --line-length 100

# Type checking
mypy surgical_theater/

# Linting
flake8 surgical_theater/ examples/ tests/
```

### Testing
```bash
# Interactive demo notebook (recommended)
jupyter notebook tests/demo.ipynb

# Run any remaining unit tests
pytest

# Run with coverage
pytest --cov=surgical_theater
```

### Examples
```bash
# Run basic usage examples
python examples/basic_usage.py

# Run LoRA integration examples
python examples/lora_integration.py

# Run advanced features demo (quantized/sharded models)
python examples/advanced_features.py
```

### Building
```bash
# Build package
python -m build

# Install from built package
pip install dist/surgical_theater-*.whl
```

## Package Structure

- **`surgical_theater/core.py`**: Main SurgicalTheater implementation with weight backup/restore logic
- **`examples/basic_usage.py`**: 6 comprehensive examples covering validation, hyperparameter testing, ablation studies
- **`examples/lora_integration.py`**: LoRA-specific examples showing memory comparisons
- **`examples/advanced_features.py`**: Demonstrates quantized model support and FSDP/DeepSpeed compatibility
- **`tests/demo.ipynb`**: Interactive Jupyter notebook demonstrating all key features and use cases
- **`.github/workflows/test.yml`**: CI/CD pipeline with matrix testing across Python/PyTorch versions
- **`pyproject.toml`**: Modern Python packaging configuration with dev/examples optional dependencies

## Recent Critical Improvements (Based on Reviewer Feedback)

The codebase has been significantly improved based on expert review:

### Fixed Issues (v0.1.0)
1. **False "zero-copy" claims**: Now uses honest delta-based approach (~1 parameter set overhead)
2. **Custom modification memory leaks**: Custom functions now return deltas directly (no double-cloning)
3. **Thread safety**: Added re-entrancy protection with `_entered` flag
4. **Tensor aliasing**: Added contiguity enforcement to prevent storage view issues
5. **Error handling**: Changed from warnings to RuntimeError for restoration failures

### Latest Improvements (v0.1.1+)
6. **Flexible re-entrancy**: Changed from strict blocking to depth=1 allowance using `_enter_depth` counter
7. **Quantized model support**: Full support for bitsandbytes/QLoRA with FP32 copy-apply-cast pattern
8. **FSDP/DeepSpeed sharding**: Automatic detection and parameter gathering for distributed models
9. **Dtype preservation**: Enhanced dtype consistency checking during delta operations

### Critical Production Fixes (v0.2.0)
10. **RAM spike prevention**: Deltas stored in original dtype (int8/int4) not FP32
11. **CPU offloading safety**: Skip gather path for CPU-offloaded bnb parameters
12. **Gradient flow preservation**: Cache/restore `requires_grad` flags via `_requires_grad_cache`
13. **Embedding layer handling**: Full-copy fallback for 2-D tensors with wrong shard shapes
14. **Multi-node stability**: All distributed ops use explicit `group=` parameter
15. **Comprehensive testing**: GitHub Actions matrix ensures memory claims stay within ±5%
16. **Training mode restoration**: Cache/restore `model.training` state via `_training_mode_cache`

### Key Implementation Details
- **Delta-based restoration**: `param.data.add_(delta)` then `param.data.sub_(delta)`
- **Quantized parameter handling**: Auto-detect → copy to FP32 → apply delta → cast back → store delta in original dtype
- **Custom int8 addition**: `_custom_add_int8_inplace()` for bitsandbytes parameters
- **Sharded parameter gathering**: FSDP/DeepSpeed detection → gather → modify → distribute back
- **CPU offload handling**: Skip gather for CPU-offloaded bnb params to prevent crashes
- **Gradient preservation**: `_cache_requires_grad()` and `_restore_requires_grad()` methods
- **Training mode preservation**: `_cache_training_mode()` and `_restore_training_mode()` methods
- **Embedding layer safety**: 2-D tensor detection with full-copy fallback
- **Distributed synchronization**: All ops use explicit `group=` and barriers
- **Shape validation**: Strict delta shape checking before application
- **Device handling**: Automatic tensor device consistency
- **Dtype preservation**: Automatic dtype casting for deltas and validation
- **Re-entrancy depth**: `_enter_depth` counter allows depth=1 nesting
- **Scalar broadcasting**: Proper `torch.as_tensor()` usage for cross-device scalars

### Custom Modification API
```python
# CORRECT: Return delta directly
def custom_fn(param, scale=2.0):
    return param * (scale - 1.0)  # Return delta

# WRONG: Modify in-place (old API)
def bad_fn(param, scale=2.0):
    param.data *= scale  # Don't do this
```

## Memory Efficiency Details

The package achieves ~2-10x memory reduction compared to `deepcopy()` by:
- Only storing deltas for parameters that will be modified (not entire model)
- Using delta-based restoration instead of full parameter copies  
- Automatic cleanup on context exit (even on exceptions)

Traditional validation: 20GB model + 20GB copy = 40GB total
SurgicalTheater validation: 20GB model + ~2GB deltas = ~22GB total

## Integration Notes

- Compatible with any PyTorch model architecture
- Works with LoRA, PEFT, full fine-tuning, and RL training
- Supports both automatic layer detection (attention layers) and manual layer specification
- Thread-safe and exception-safe through proper context manager implementation
- **Quantized models**: Full support for bitsandbytes/QLoRA with automatic FP32 conversion
- **Sharded models**: Compatible with FSDP, DeepSpeed, and HuggingFace Accelerate
- Handles compiled models and non-contiguous tensors with automatic contiguity enforcement
- Auto-detection uses deterministic ordering for reproducible results
- Allows depth=1 re-entrancy for common nested usage patterns

## Critical Architecture Components (v0.1.1+)

### Quantization Support
- `_is_quantized_parameter()`: Detects bitsandbytes/QLoRA parameters
- `_quantized_params`: Tracks original dtypes for restoration
- Copy-as-FP32 → apply delta → cast-back pattern prevents quantization corruption

### Sharding Support  
- `_detect_sharded_model()`: Auto-detects FSDP/DeepSpeed/Accelerate sharding
- `_gather_sharded_parameter()`: Gathers distributed parameters for modification
- `_restore_sharded_parameter()`: Distributes deltas back to shards correctly
- `_sharded_params`: Tracks sharding metadata for proper restoration

### Memory Efficiency
- Delta storage uses ~1 parameter set vs 2x full model for deepcopy
- Quantized models: Deltas stored in original dtype (int8/int4) to prevent RAM spikes
- Automatic cleanup on context exit (even with exceptions)
- Memory tracking with `total_delta_memory_mb` property for monitoring
- CI/CD enforces memory usage stays within ±5% of README claims

### Testing Strategy
- **Demo notebook**: Interactive `tests/demo.ipynb` showcasing all key features and use cases
- **Core functionality**: Memory efficiency, state restoration, gradient preservation
- **Advanced features**: Quantized models, sharded models, custom modifications, exception safety
- **Training integration**: Real-world usage patterns and validation workflows
- **GitHub Actions**: Matrix testing across Python 3.8-3.11, PyTorch 2.0-2.2, with bitsandbytes
- **Memory verification**: Automated checks that delta memory matches ~1 parameter set claim

### Demo Notebook Features
The `tests/demo.ipynb` notebook demonstrates:
1. **Memory-efficient validation** with ResNet-18 showing dramatic memory savings
2. **Perfect state restoration** with weight and gradient flag preservation
3. **Custom modification functions** for specialized use cases
4. **Exception safety** ensuring robust error handling
5. **Training integration** showing real-world usage patterns

This notebook serves as both documentation and verification of SurgicalTheater's capabilities.