"""
SurgicalTheater: Memory-efficient context manager for temporary model modifications.

This module provides memory-efficient temporary modifications to PyTorch models,
enabling safe validation and experimentation with minimal memory overhead.
Uses delta-based approach instead of full weight cloning.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, List, Optional, Union, Callable
from collections import defaultdict
import warnings


class SurgicalTheater:
    """
    Context manager for temporary, memory-efficient model modifications.
    
    Provides delta-based weight modifications with automatic restoration,
    enabling safe model experimentation with minimal memory overhead.
    Uses ~1 parameter set of extra memory instead of 2x full model cloning.
    
    Examples:
        Basic usage:
        >>> with SurgicalTheater(model) as theater:
        ...     # Model is temporarily modified here
        ...     val_loss = model(validation_data)
        ... # Model automatically restored
        
        With specific modifications:
        >>> with SurgicalTheater(model, layers=[0, 1], modification_type="scale", factor=0.9) as theater:
        ...     # factor=0.9 → delta = -10% (reduces weights by 10%)
        ...     performance = evaluate(model)
    """
    
    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        modification_type: str = "custom",
        modification_fn: Optional[Callable] = None,
        track_memory: bool = False,
        **kwargs
    ):
        """
        Initialize SurgicalTheater context manager.
        
        Args:
            model: PyTorch model to temporarily modify
            layers: List of layer indices to modify (None = auto-detect)
            modification_type: Type of modification ("custom", "scale", "noise", "disable")
            modification_fn: Custom modification function
            track_memory: Whether to track memory usage
            **kwargs: Additional arguments for modification functions
        """
        self.model = model
        self.layers = layers
        self.modification_type = modification_type
        self.modification_fn = modification_fn
        self.track_memory = track_memory
        self.kwargs = kwargs
        
        # Storage for deltas (memory-efficient approach)
        self._deltas = {}
        self._target_params = {}
        self._quantized_params = {}  # Track which params were quantized
        self._sharded_params = {}  # Track sharded parameter info
        self._requires_grad_cache = {}  # Cache requires_grad flags
        
        # Memory tracking
        self._memory_before = 0
        self._memory_saved = 0
        
        # Modification tracking
        self._modifications_applied = []
        
        # Thread safety / re-entrancy protection (allow depth=1)
        self._enter_depth = 0
    
    def __enter__(self):
        """Enter context and apply temporary modifications."""
        # Check for nested re-entrancy (allow depth=1)
        self._enter_depth = getattr(self, "_enter_depth", 0) + 1
        if self._enter_depth > 1:
            self._enter_depth -= 1  # Reset on error
            raise RuntimeError("Nested SurgicalTheater instances on the same object aren't supported. "
                               "Use depth-1 or different theatre objects.")
        
        try:
            if self.track_memory:
                self._memory_before = self._get_memory_usage()
            
            # Identify target parameters
            self._identify_target_parameters()
            
            # Cache requires_grad flags before modification
            self._cache_requires_grad()
            
            # Ensure tensor contiguity for safe delta operations
            self._ensure_contiguity()
            
            # Compute and apply deltas
            self._apply_deltas()
            
            return self
        except Exception:
            self._enter_depth -= 1  # Reset on error
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original weights."""
        try:
            # Restore by subtracting deltas
            self._restore_from_deltas()
            
            # Restore requires_grad flags
            self._restore_requires_grad()
        except Exception as e:
            raise RuntimeError(f"Failed to restore model weights: {e}") from e
        finally:
            if self.track_memory:
                memory_after = self._get_memory_usage()
                self._memory_saved = max(0, self._memory_before - memory_after)
            
            # Only clear storage when fully exiting (depth=0)
            self._enter_depth -= 1
            if self._enter_depth == 0:
                self._deltas.clear()
                self._target_params.clear()
                self._quantized_params.clear()
                self._sharded_params.clear()
                self._requires_grad_cache.clear()
                self._modifications_applied.clear()
    
    def _identify_target_parameters(self):
        """Identify parameters to modify based on layers specification."""
        target_layers = self._get_target_layers()
        
        # Check if model is sharded (FSDP/DeepSpeed)
        model_is_sharded = self._detect_sharded_model()
        
        for layer_name, module in self.model.named_modules():
            if self._should_modify_layer(layer_name, module, target_layers):
                # Add parameters that will be modified
                for param_name, param in module.named_parameters(recurse=False):
                    # Include both trainable and frozen parameters for broader compatibility
                    if param is not None:
                        key = f"{layer_name}.{param_name}"
                        
                        # Handle sharded parameters
                        if model_is_sharded:
                            gathered_param = self._gather_sharded_parameter(param, key)
                            if gathered_param is not None:
                                self._target_params[key] = gathered_param
                                self._sharded_params[key] = {
                                    'original_param': param,
                                    'is_sharded': True
                                }
                            else:
                                # Skip parameters that couldn't be gathered
                                continue
                        else:
                            self._target_params[key] = param
    
    def _cache_requires_grad(self):
        """Cache requires_grad flags for all target parameters."""
        for param_key, param in self._target_params.items():
            self._requires_grad_cache[param_key] = param.requires_grad
    
    def _restore_requires_grad(self):
        """Restore requires_grad flags for all target parameters."""
        for param_key, param in self._target_params.items():
            if param_key in self._requires_grad_cache:
                param.requires_grad = self._requires_grad_cache[param_key]
    
    def _ensure_contiguity(self):
        """Ensure all target parameters are contiguous to avoid storage aliasing issues."""
        for param_key, param in self._target_params.items():
            if not param.data.is_contiguous():
                # Force contiguity to avoid view/transpose/compile aliasing issues
                param.data = param.data.contiguous()
    
    def _apply_deltas(self):
        """Compute and apply deltas to target parameters."""
        for param_key, param in self._target_params.items():
            # Handle quantized parameters with copy-as-fp32 → apply delta → cast-back pattern
            is_quantized = self._is_quantized_parameter(param)
            
            if is_quantized:
                # Store quantization info for restoration
                self._quantized_params[param_key] = {
                    'original_dtype': param.dtype,
                    'original_device': param.device
                }
                
                # Copy to FP32 for delta computation and application
                param_fp32 = param.data.float()
                
                # Compute delta on FP32 version
                delta_fp32 = self._compute_delta_for_param(param_fp32, param_key)
                
                # Apply delta to FP32 version
                modified_fp32 = param_fp32 + delta_fp32
                
                # Cast back to original quantized format and update parameter
                param.data = modified_fp32.to(param.dtype)
                
                # Store delta in original dtype to prevent RAM spike
                delta = delta_fp32.to(param.dtype)
                self._deltas[param_key] = delta.detach()
            else:
                # Standard non-quantized parameter handling
                original_dtype = param.dtype
                
                # Compute delta based on modification type
                delta = self._compute_delta_for_param(param.data, param_key)
                
                # Validate delta shape and compatibility
                if delta.shape != param.shape:
                    raise ValueError(f"Delta shape {delta.shape} doesn't match param shape {param.shape} for {param_key}")
                
                # Ensure delta is on same device and dtype as parameter
                if delta.device != param.device:
                    delta = delta.to(param.device)
                if delta.dtype != param.dtype:
                    delta = delta.to(param.dtype)
                
                # Store delta for restoration (this is our memory-efficient approach)
                self._deltas[param_key] = delta.detach()
                
                # Apply delta in-place
                param.data.add_(delta)
                
                # Verify dtype preservation
                if param.dtype != original_dtype:
                    raise RuntimeError(f"Parameter {param_key} dtype changed from {original_dtype} to {param.dtype} "
                                     "during modification. This indicates a dtype consistency issue.")
            
            self._modifications_applied.append({
                'param': param_key,
                'type': self.modification_type,
                'delta_norm': self._deltas[param_key].norm().item(),
                'is_quantized': is_quantized
            })
    
    def _restore_from_deltas(self):
        """Restore original parameters by subtracting deltas."""
        for param_key, delta in self._deltas.items():
            if param_key in self._target_params:
                param = self._target_params[param_key]
                
                # Handle sharded parameter restoration
                if param_key in self._sharded_params:
                    self._restore_sharded_parameter(param_key, delta)
                    continue
                
                # Handle quantized parameter restoration
                if param_key in self._quantized_params:
                    # For quantized params: copy to FP32, subtract delta, cast back
                    param_fp32 = param.data.float()
                    restored_fp32 = param_fp32 - delta
                    param.data = restored_fp32.to(self._quantized_params[param_key]['original_dtype'])
                else:
                    # Standard restoration by subtracting the delta we applied
                    param.data.sub_(delta)
            else:
                raise RuntimeError(f"Parameter {param_key} not found during restoration")
    
    def _restore_sharded_parameter(self, param_key: str, delta: torch.Tensor):
        """Restore a sharded parameter by distributing the delta back."""
        try:
            sharded_info = self._sharded_params[param_key]
            original_param = sharded_info['original_param']
            
            # For FSDP, we need to distribute the delta back to the sharded parameter
            if hasattr(original_param, '_local_shard'):
                # Get the local shard portion of the delta
                # This is a simplified approach - production code may need more sophisticated sharding logic
                world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                
                if world_size > 1:
                    # Get the default process group for safety
                    group = torch.distributed.group.WORLD
                    
                    # Calculate shard boundaries
                    total_elements = delta.numel()
                    elements_per_shard = total_elements // world_size
                    start_idx = rank * elements_per_shard
                    end_idx = start_idx + elements_per_shard if rank < world_size - 1 else total_elements
                    
                    # Extract local shard from delta and apply
                    delta_flat = delta.flatten()
                    local_delta = delta_flat[start_idx:end_idx].reshape(original_param.shape)
                    original_param.data.sub_(local_delta)
                    
                    # Add barrier to ensure all ranks complete restoration
                    torch.distributed.barrier(group=group)
                else:
                    # Single device - apply full delta
                    original_param.data.sub_(delta)
            else:
                # Non-FSDP sharded parameter - apply delta directly
                original_param.data.sub_(delta)
                
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to restore sharded parameter {param_key}: {e}. Parameter may not be fully restored.")
    
    def _custom_add_int8_inplace(self, param: torch.Tensor, delta: torch.Tensor) -> None:
        """Custom in-place addition for int8 parameters (bitsandbytes)."""
        try:
            # Check if this is a bitsandbytes parameter
            if hasattr(param, '__class__') and 'Int8' in param.__class__.__name__:
                # For bitsandbytes int8 parameters, we need special handling
                # Convert to float, add delta, then quantize back
                param_fp32 = param.data.float()
                delta_fp32 = delta.float()
                result_fp32 = param_fp32 + delta_fp32
                
                # Quantize back to int8 range
                param.data = result_fp32.clamp(-127, 127).to(torch.int8)
            else:
                # Standard int8 addition with clamping
                param_fp32 = param.data.float()
                delta_fp32 = delta.float()
                result_fp32 = param_fp32 + delta_fp32
                param.data = result_fp32.clamp(-127, 127).to(param.dtype)
        except Exception as e:
            # Fallback to standard addition if custom logic fails
            param.data = param.data.float()
            param.data.add_(delta.float())
            param.data = param.data.to(param.dtype)
    
    def _is_quantized_parameter(self, param: torch.Tensor) -> bool:
        """Check if a parameter is quantized (bitsandbytes, QLoRA, etc.)."""
        # Check for integer dtypes (common in quantized models)
        if hasattr(param, 'dtype') and 'int' in str(param.dtype):
            return True
        
        # Check for bitsandbytes quantized parameters
        if hasattr(param, '__class__'):
            class_name = param.__class__.__name__
            if 'Int' in class_name or 'Quant' in class_name or 'Bits' in class_name:
                return True
        
        # Check for specific quantized tensor types
        if hasattr(param, '_quantized_param') or hasattr(param, 'quant_state'):
            return True
            
        return False
    
    def _compute_delta_for_param(self, param_data: torch.Tensor, param_key: str) -> torch.Tensor:
        """Compute delta for a specific parameter tensor."""
        return self._compute_delta(param_data)
    
    def _detect_sharded_model(self) -> bool:
        """Detect if the model is using FSDP or DeepSpeed sharding."""
        # Check for FSDP wrapper
        if hasattr(self.model, '_fsdp_wrapped_module'):
            return True
            
        # Check for DeepSpeed wrapper
        if hasattr(self.model, 'module') and hasattr(self.model, 'engine'):
            return True
            
        # Check for HuggingFace Accelerate sharding
        if hasattr(self.model, '_hf_hook') or hasattr(self.model, 'hf_device_map'):
            return True
            
        # Check for distributed model wrapper
        model_class_name = self.model.__class__.__name__
        if any(wrapper in model_class_name for wrapper in ['FSDP', 'DeepSpeed', 'DistributedDataParallel']):
            return True
            
        # Check individual parameters for sharding indicators
        for param in self.model.parameters():
            if hasattr(param, '_local_shard') or hasattr(param, '_sharded_tensor'):
                return True
                
        return False
    
    def _gather_sharded_parameter(self, param: torch.Tensor, param_key: str) -> torch.Tensor:
        """Gather a sharded parameter across devices if needed."""
        try:
            # Check for CPU-offloaded bitsandbytes parameters
            if hasattr(param, '__class__') and 'Int8' in param.__class__.__name__:
                if param.device.type == 'cpu':
                    # Skip gather path for CPU-offloaded bnb parameters
                    import warnings
                    warnings.warn(f"Skipping gather for CPU-offloaded bitsandbytes parameter {param_key}. "
                                "Using direct modification (may increase RAM usage).")
                    return param.data
            
            # Try FSDP gathering
            if hasattr(param, '_local_shard'):
                # FSDP parameter - need to gather
                if hasattr(torch.distributed.fsdp, 'FullyShardedDataParallel'):
                    # Use FSDP's summon_full_params context manager
                    parent_module = self._get_parameter_parent_module(param_key)
                    if parent_module and hasattr(parent_module, '_fsdp_wrapped_module'):
                        with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(parent_module):
                            return param.data.clone()
            
            # Try DeepSpeed gathering
            if hasattr(self.model, 'engine') and hasattr(self.model.engine, 'gather_16bit_weights_on_model_save'):
                # DeepSpeed Zero optimizer - gather weights
                # This is a simplified approach - may need refinement for production
                return param.data.clone()
            
            # Try manual gathering for other distributed setups
            if torch.distributed.is_initialized() and param.dim() > 0:
                # Check for 2-D weight tensors (embedding layers) - use full-copy fallback
                if param.dim() == 2 and param.shape[0] > param.shape[1]:
                    # Likely an embedding layer - shards are cut across rows
                    import warnings
                    warnings.warn(f"<WARN: full-copy used> for 2-D tensor {param_key} with shape {param.shape}. "
                                "Embedding layer sharding not fully supported yet.")
                    # Use full copy for now - TODO: implement proper embedding sharding
                    return param.data.clone()
                
                # Simple all-gather for distributed parameters
                world_size = torch.distributed.get_world_size()
                if world_size > 1:
                    # Get the default process group for safety
                    group = torch.distributed.group.WORLD
                    gathered_tensors = [torch.zeros_like(param) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered_tensors, param, group=group)
                    
                    # Add barrier to ensure all ranks complete gathering
                    torch.distributed.barrier(group=group)
                    
                    # Concatenate or select based on sharding strategy
                    return torch.cat(gathered_tensors, dim=0)
            
            # If no sharding detected, return as-is
            return param.data
            
        except Exception as e:
            # If gathering fails, warn and skip this parameter
            import warnings
            warnings.warn(f"Failed to gather sharded parameter {param_key}: {e}. Skipping this parameter.")
            return None
    
    def _get_parameter_parent_module(self, param_key: str):
        """Get the parent module for a parameter key."""
        parts = param_key.split('.')
        if len(parts) < 2:
            return self.model
            
        # Navigate to parent module
        module = self.model
        for part in parts[:-1]:  # Exclude the parameter name
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def _compute_delta(self, param: torch.Tensor) -> torch.Tensor:
        """Compute delta for a parameter based on modification type."""
        if self.modification_type == "scale":
            factor = self.kwargs.get('factor', 0.9)
            # Ensure factor is a tensor on the correct device
            factor = torch.as_tensor(factor, device=param.device, dtype=param.dtype)
            # delta = param * factor - param = param * (factor - 1)
            return param * (factor - 1.0)
        
        elif self.modification_type == "noise":
            noise_scale = self.kwargs.get('noise_scale', 0.01)
            # Ensure noise_scale is a tensor on the correct device
            noise_scale = torch.as_tensor(noise_scale, device=param.device, dtype=param.dtype)
            # delta = noise
            return torch.randn_like(param) * noise_scale
        
        elif self.modification_type == "disable":
            # delta = 0 - param = -param
            return -param
        
        elif self.modification_type == "custom" and self.modification_fn:
            # For custom modifications, require function to return delta directly
            # This avoids double-cloning which negates memory benefits
            delta = self.modification_fn(param, **self.kwargs)
            if delta is None:
                raise ValueError("Custom modification function must return a delta tensor")
            # Shape checking will be done in _apply_deltas
            return delta
        
        else:
            raise ValueError(f"Unknown modification type: {self.modification_type}")
    
    def _get_target_layers(self) -> List[str]:
        """Get list of target layer names."""
        if self.layers is None:
            # Auto-detect attention layers (deterministic ordering)
            target_layers = []
            attention_keywords = ['attention', 'attn', 'self_attn']  # Deterministic order
            for name, module in self.model.named_modules():
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in attention_keywords):
                    target_layers.append(name)
            # Sort for deterministic ordering across runs
            return sorted(target_layers)
        
        # Convert layer indices to names
        all_layers = list(self.model.named_modules())
        target_layers = []
        
        for idx in self.layers:
            if 0 <= idx < len(all_layers):
                layer_name, _ = all_layers[idx]
                target_layers.append(layer_name)
        
        return target_layers
    
    
    def _should_modify_layer(self, layer_name: str, module: nn.Module, target_layers: List[str]) -> bool:
        """Check if layer should be modified."""
        return layer_name in target_layers
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get module by its name."""
        if name == '':
            return self.model
        
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module
    
    def _get_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0
    
    @property
    def memory_saved(self) -> float:
        """Get memory saved in GB."""
        return self._memory_saved / (1024 ** 3)
    
    @property
    def modifications_summary(self) -> Dict:
        """Get summary of modifications applied."""
        summary = defaultdict(list)
        for mod in self._modifications_applied:
            summary[mod['type']].append(mod['param'])
        return dict(summary)
    
    @property
    def delta_statistics(self) -> Dict:
        """Get statistics about the deltas applied."""
        stats = {}
        for param_key, delta in self._deltas.items():
            stats[param_key] = {
                'shape': tuple(delta.shape),
                'mean_abs_delta': delta.abs().mean().item(),
                'max_abs_delta': delta.abs().max().item(),
                'memory_mb': delta.numel() * delta.element_size() / (1024 * 1024)
            }
        return stats
    
    @property
    def total_delta_memory_mb(self) -> float:
        """Get total memory used by all deltas in MB."""
        return sum(delta.numel() * delta.element_size() for delta in self._deltas.values()) / (1024 * 1024)


# Convenience function
def surgical_theater(
    model: nn.Module,
    layers: Optional[List[int]] = None,
    modification_type: str = "scale",
    **kwargs
) -> SurgicalTheater:
    """
    Convenience function to create SurgicalTheater context manager.
    
    Args:
        model: PyTorch model to modify
        layers: Layer indices to modify
        modification_type: Type of modification
        **kwargs: Additional modification parameters
    
    Returns:
        SurgicalTheater context manager
    
    Example:
        >>> with surgical_theater(model, layers=[0, 1], factor=0.9):
        ...     val_loss = model(data)
    """
    return SurgicalTheater(model, layers, modification_type, **kwargs)