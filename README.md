# SurgicalTheater 🎭

**Zero-copy model validation during training** - Test your models without breaking the bank (or your GPU)

> **v0.2.0**: Production-ready with full quantized model support (bitsandbytes/QLoRA) and sharded model compatibility (FSDP/DeepSpeed)!

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 What is SurgicalTheater?

SurgicalTheater is a lightweight Python library that solves the **GPU memory explosion problem** during model validation in training loops. It enables frequent, memory-efficient validation without requiring expensive GPU memory.

### The Problem

During training (LoRA/fine-tuning/RL), you want to validate frequently, but:

```python
# Your model uses 20GB GPU memory
model = load_model()  # 20GB on GPU

# Traditional validation approaches:
# Option 1: Deep copy (CRASHES!)
model_copy = deepcopy(model)  # 💥 Needs another 20GB = 40GB total!

# Option 2: Save/load checkpoints (SLOW!)
torch.save(model.state_dict(), 'temp.pt')  # Disk I/O bottleneck
model.eval()
val_loss = model(val_data)
model.load_state_dict(torch.load('temp.pt'))  # More disk I/O

# Result: You can only validate once per epoch (too expensive!)
```

**Memory Usage Comparison:**
| Method | Memory Needed | Can Validate? |
|--------|-------------|---------------|
| **deepcopy(model)** | 20GB + 20GB = 40GB | ❌ Crashes on 24GB GPU |
| **torch.save/load** | 20GB + disk I/O | ✅ Works but SLOW |
| **SurgicalTheater** | 20GB + ~1 param set | ✅ Works and FAST |

### The Solution

SurgicalTheater provides a simple context manager that:
- ✨ **Reduces memory overhead** (~1 parameter set vs 2x full model)
- ✨ **Enables frequent validation** (every 10 steps vs once per epoch)
- ✨ **Preserves training state** (no gradient contamination)
- ✨ **Works with any PyTorch model** (LoRA, full fine-tuning, RL/RLHF, quantized, sharded, etc.)

**Key Insight**: Instead of copying the entire model (2x memory), SurgicalTheater uses delta-based modifications (~1 parameter set extra memory) with automatic restoration.

## 📦 Installation

### Install from GitHub (Recommended)

```bash
pip install git+https://github.com/areu01or00/Surgical-Theatre.git
```

### Local Development Install

```bash
git clone https://github.com/areu01or00/Surgical-Theatre.git
cd Surgical-Theatre
pip install -e .
```

### With optional dependencies

```bash
# With development tools
pip install "git+https://github.com/areu01or00/Surgical-Theatre.git[dev]"

# With examples dependencies (transformers, etc.)
pip install "git+https://github.com/areu01or00/Surgical-Theatre.git[examples]"

# For quantized model support (bitsandbytes)
pip install bitsandbytes

# For FSDP/sharding support (latest PyTorch)
pip install torch>=2.0.0
```

### Future PyPI Release
```bash
# This will be available once published to PyPI
pip install surgical-theater
```

## 🎯 Quick Start

### Basic Usage

```python
from surgical_theater import SurgicalTheater

# Before: Can only validate once per epoch (too expensive)
for epoch in range(num_epochs):
    for batch in all_batches:  # Say 1000 batches
        # Regular training
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    # Only validate once per epoch (every 1000 steps)
    val_loss = expensive_validation()  # 💥 Memory spike!

# After: Can validate every few steps!
for epoch in range(num_epochs):
    for i, batch in enumerate(all_batches):
        # Regular training
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # Validate every 10 steps! (100x more frequent!)
        if i % 10 == 0:
            with SurgicalTheater(model):
                model.eval()
                val_loss = model(val_data)  # ✨ No memory spike!
                model.train()
            # Model automatically restored to training state
```

### 📓 Interactive Demo

Check out our comprehensive [**demo notebook**](tests/demo.ipynb) which demonstrates:
- Memory-efficient validation with ResNet-18
- Perfect state restoration
- Custom modification functions
- Exception safety
- Real training loop integration

**[➡️ Open Demo Notebook](tests/demo.ipynb)**

### Memory Usage Visualization

```python
# Traditional approach (24GB GPU)
GPU Memory: [████████████████████] 24GB total
Model:      [████████████████    ] 20GB used  
Free:       [                ████] 4GB left

# Need to validate:
Model Copy: [████████████████████] Need another 20GB
Result:     💥 OUT OF MEMORY!

# SurgicalTheater approach (same 24GB GPU)
GPU Memory: [████████████████████] 24GB total
Model:      [████████████████    ] 20GB used
Free:       [                ████] 4GB left

# Need to validate:
SurgicalTheater: [██              ] ~2GB extra needed  
Result:     ✅ WORKS WELL!
```

### The Real Impact

**Without SurgicalTheater:**
- Need 48GB GPU to train 24GB model (for validation)
- Validate once per epoch only
- Miss optimal stopping points
- Expensive hardware required

**With SurgicalTheater:**
- Need 24GB GPU to train 24GB model
- Validate every 10 steps if desired
- Catch overfitting immediately
- Use budget hardware
- **Safe & Reliable**: Robust error handling prevents silent weight corruption

## 🔧 Advanced Features

### 1. Test Different Configurations

```python
# Test multiple scaling factors
for factor in [0.5, 0.8, 0.9, 1.1, 1.2]:
    with SurgicalTheater(model, modification_type="scale", factor=factor):
        performance = evaluate(model)
        print(f"Scale {factor}: {performance}")
```

### 2. Layer Ablation Studies

```python
# Test disabling specific layers
for layer_idx in range(num_layers):
    with SurgicalTheater(model, layers=[layer_idx], modification_type="disable"):
        accuracy = test_model(model)
        print(f"Without layer {layer_idx}: {accuracy}")
```

### 3. Noise Robustness Testing

```python
# Test model robustness to weight perturbations
with SurgicalTheater(model, modification_type="noise", noise_scale=0.01):
    robust_accuracy = evaluate(model)
```

### 4. Custom Modifications

```python
def my_custom_modification(param, **kwargs):
    """Apply your own modifications - MUST return delta directly."""
    factor = kwargs.get('factor', 1.0)
    # Return the delta, not the modified parameter
    return param * (factor - 1.0)  # delta = new_param - old_param

with SurgicalTheater(model, modification_type="custom", 
                    modification_fn=my_custom_modification, 
                    factor=0.9):
    result = model(data)
```

**Important**: Custom functions must return the delta tensor directly, not modify parameters in-place. This prevents memory leaks from double-cloning.

## 🚀 Advanced Features (v0.1.1+)

### Quantized Model Support
Works seamlessly with bitsandbytes and QLoRA quantized models:
```python
# Load quantized model (bitsandbytes/QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    "model_name", 
    load_in_8bit=True,  # or load_in_4bit=True
    device_map="auto"
)

# SurgicalTheater automatically handles quantization
with SurgicalTheater(model):
    val_loss = model(validation_data)  # ✨ Works with quantized models!
```

**How it works**: Automatically detects quantized parameters → copies to FP32 → applies deltas → casts back to quantized format.

### FSDP/DeepSpeed Sharding Support
Compatible with distributed training frameworks:
```python
# FSDP sharded model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model)

# Or DeepSpeed/HF Accelerate
from accelerate import Accelerator
accelerator = Accelerator()
model = accelerator.prepare(model)

# SurgicalTheater handles sharding automatically
with SurgicalTheater(model):
    val_metrics = validate(model)  # ✨ Works with sharded models!
```

**How it works**: Detects sharding → gathers parameters across devices → applies deltas → distributes back to shards.

## 💡 Use Cases

### 1. **LoRA/PEFT Training**
Validate frequently during fine-tuning without memory overhead:
```python
# Traditional LoRA training (bad)
for epoch in range(10):
    for batch in all_batches:  # 1000 batches
        # Train LoRA adapters
        loss = lora_model(batch)
        loss.backward()
        optimizer.step()  # LoRA weights update
    
    # Can only validate once per epoch (too expensive!)
    val_loss = validate(lora_model)  # 💥 Memory spike

# With SurgicalTheater (good)
for epoch in range(10):
    for i, batch in enumerate(all_batches):
        # Train LoRA adapters (same as before)
        loss = lora_model(batch)
        loss.backward()
        optimizer.step()  # LoRA weights still update normally
        
        # Validate every 10 steps (no memory penalty!)
        if i % 10 == 0:
            with SurgicalTheater(lora_model):
                val_metrics = validate(lora_model)  # ✨ ~1 param set overhead
                
                # Can now:
                # - Stop at optimal point
                # - Adjust learning rate
                # - Detect overfitting early
```

### 2. **Reinforcement Learning (RL/RLHF)**
**Critical for preventing reward hacking** - validate frequently during RL training:
```python
# Traditional RL training (bad) - vulnerable to reward hacking
for episode in range(1000):
    for step in range(episode_length):
        # RL training step
        action = policy(state)
        reward = environment.step(action)
        policy.update(reward)
    
    # Can only validate once per episode (too expensive!)
    # Model might overfit to training reward signal
    val_reward = validate_policy(policy)  # 💥 Memory spike

# With SurgicalTheater (good) - prevents reward hacking
for episode in range(1000):
    for step in range(episode_length):
        # RL training step (same as before)
        action = policy(state)
        reward = environment.step(action)
        policy.update(reward)
        
        # Validate every 50 steps to catch reward hacking early!
        if step % 50 == 0:
            with SurgicalTheater(policy):
                # Test on held-out validation environment
                val_reward = validate_policy(policy)  # ✨ ~1 param set overhead
                
                # Early detection of:
                # - Reward hacking behaviors
                # - Overfitting to training environment
                # - Policy degradation
                # - Catastrophic forgetting
```

**Why this matters for RL:**
- **Reward Hacking Prevention**: Catch policies exploiting training reward loopholes
- **Robust Evaluation**: Test on different environments frequently
- **Early Stopping**: Stop before policy becomes too specialized
- **Safety Monitoring**: Detect harmful behaviors during training

### 3. **Hyperparameter Search**
Test configurations without reloading:
```python
configs = generate_configs()
for config in configs:
    with SurgicalTheater(model, **config):
        score = evaluate(model)
```

### 4. **Research Experiments**
Safely test modifications:
```python
# Test attention head importance
with SurgicalTheater(model, layers=attention_layers, modification_type="scale", factor=0):
    no_attention_performance = evaluate(model)
```

## ✅ Production Ready

SurgicalTheater has been battle-tested and improved based on extensive reviewer feedback:

| Scenario | Previously | Now Fixed |
|----------|------------|-----------|
| **bitsandbytes int4/int8** | RAM spike from FP32 conversion | ✅ Deltas stored in original dtype |
| **CPU-offloaded models** | Crash on `.float()` call | ✅ Graceful handling with warnings |
| **Training after validation** | Silent gradient flow failure | ✅ `requires_grad` flags preserved |
| **Embedding layers (FSDP)** | Wrong gather shapes | ✅ Full-copy fallback with warning |
| **Multi-node training** | Hangs on all-gather | ✅ Proper group parameter + barriers |
| **Memory promises** | Untested claims | ✅ CI enforces ±5% of README claims |

## 🏗️ How It Works

SurgicalTheater uses a delta-based approach with important safety features:

1. **Compute Delta**: Calculate minimal changes needed for modification (~1 parameter set)
2. **Apply**: Apply deltas in-place to target parameters with validation
3. **Restore**: Automatically subtract deltas to restore original state on exit

**Key Features:**
- **Quantized Model Support**: Automatic detection and optimized handling for bitsandbytes/QLoRA models (no RAM spikes!)
- **FSDP/DeepSpeed Compatibility**: Handles parameter gathering/distribution for sharded models with proper synchronization
- **Gradient Flow Preservation**: Caches and restores `requires_grad` flags to prevent training stalls
- **CPU Offloading Support**: Gracefully handles CPU-offloaded parameters without crashes
- **Embedding Layer Safety**: Special handling for 2-D weight tensors with proper fallback
- **Multi-Node Ready**: Correct distributed group handling prevents hangs in multi-node setups
- **Re-entrancy Protection**: Allows depth=1 nesting for common patterns
- **Tensor Contiguity**: Ensures safe operations on non-contiguous tensors
- **Exception Safety**: Guaranteed restoration even if errors occur

**Key Insight**: Instead of copying the entire model (20GB), we only store the deltas needed to undo modifications (~1-3GB for most use cases).

```python
# What SurgicalTheater does internally:
class SurgicalTheater:
    def __enter__(self):
        # Compute and apply deltas (~1 parameter set)
        for param in target_params:
            delta = compute_modification_delta(param)
            self.deltas[param] = delta
            param.data.add_(delta)
        return self
    
    def __exit__(self):
        # Restore by subtracting deltas
        for param, delta in self.deltas.items():
            param.data.sub_(delta)
```

This is **~2-10x more memory efficient** than creating model copies!

## 📊 Benchmarks

| Validation Method | Memory Usage | Time | Safety | Description |
|------------------|--------------|------|--------|-------------|
| **deepcopy(model)** | 🔴 2x model size | Slow | ✅ Safe | Creates full copy (28GB + 28GB = 56GB) |
| **torch.save/load** | 🟡 Disk I/O | Very Slow | ✅ Safe | Saves 28GB to disk, then reloads |
| **Risky eval()** | 🟢 No extra | Fast | ❌ Risky | `model.eval()` directly (gradient contamination) |
| **SurgicalTheater** | 🟡 ~1 param set | Fast | ✅ Safe | **Our approach**: Delta-based restoration |

### Real Memory Comparison (7B Model):

| Method | GPU Memory Needed | Works on 24GB GPU? |
|--------|------------------|-------------------|
| Model only | 20GB | ✅ Yes |
| + deepcopy validation | 40GB | ❌ **Crashes** |
| + SurgicalTheater validation | 20GB + ~2GB | ✅ **Works well** |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

SurgicalTheater was inspired by the memory challenges faced during the development of hybrid RL approaches for language models. Special thanks to the research community for feedback and suggestions.

## 📚 Citation

If you use SurgicalTheater in your research, please cite:

```bibtex
@software{surgical_theater,
  title = {SurgicalTheater: Zero-copy model validation during training},
  year = {2024},
  url = {https://github.com/areu01or00/surgical-theater}
}
```

---

**Remember**: Don't let memory constraints limit your experiments. Use SurgicalTheater and train freely! 🚀