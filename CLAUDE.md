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
# Run all tests
pytest

# Run with coverage
pytest --cov=surgical_theater

# Run specific test file
pytest tests/test_core.py

# Run specific test function
pytest tests/test_core.py::test_surgical_theater_basic
```

### Examples
```bash
# Run basic usage examples
python examples/basic_usage.py

# Run LoRA integration examples
python examples/lora_integration.py
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
- **`pyproject.toml`**: Modern Python packaging configuration with dev/examples optional dependencies

## Recent Critical Improvements (Based on Reviewer Feedback)

The codebase has been significantly improved based on expert review:

### Fixed Issues (v0.1.0)
1. **False "zero-copy" claims**: Now uses honest delta-based approach (~1 parameter set overhead)
2. **Custom modification memory leaks**: Custom functions now return deltas directly (no double-cloning)
3. **Thread safety**: Added re-entrancy protection with `_entered` flag
4. **Tensor aliasing**: Added contiguity enforcement to prevent storage view issues
5. **Error handling**: Changed from warnings to RuntimeError for restoration failures

### Latest Improvements (v0.1.1)
6. **Flexible re-entrancy**: Changed from strict blocking to depth=1 allowance using `_enter_depth` counter
7. **Quantization detection**: Added runtime error for quantized models with helpful error messages
8. **Dtype preservation**: Enhanced dtype consistency checking during delta operations

### Key Implementation Details
- **Delta-based restoration**: `param.data.add_(delta)` then `param.data.sub_(delta)`
- **Shape validation**: Strict delta shape checking before application
- **Device handling**: Automatic tensor device consistency
- **Dtype preservation**: Automatic dtype casting for deltas and validation
- **Re-entrancy depth**: `_enter_depth` counter allows depth=1 nesting
- **Quantization safety**: Runtime detection and blocking of quantized parameters
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
- **Quantized models**: Explicitly detected and blocked with helpful error messages (use unquantized for validation)
- Handles compiled models and non-contiguous tensors with automatic contiguity enforcement
- Auto-detection uses deterministic ordering for reproducible results
- Allows depth=1 re-entrancy for common nested usage patterns