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
        
        # Memory tracking
        self._memory_before = 0
        self._memory_saved = 0
        
        # Modification tracking
        self._modifications_applied = []
        
        # Thread safety / re-entrancy protection
        self._entered = False
    
    def __enter__(self):
        """Enter context and apply temporary modifications."""
        # Check for re-entrancy
        if self._entered:
            raise RuntimeError("SurgicalTheater is not re-entrant. Nested contexts are not supported.")
        self._entered = True
        
        try:
            if self.track_memory:
                self._memory_before = self._get_memory_usage()
            
            # Identify target parameters
            self._identify_target_parameters()
            
            # Ensure tensor contiguity for safe delta operations
            self._ensure_contiguity()
            
            # Compute and apply deltas
            self._apply_deltas()
            
            return self
        except Exception:
            self._entered = False
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original weights."""
        try:
            # Restore by subtracting deltas
            self._restore_from_deltas()
        except Exception as e:
            raise RuntimeError(f"Failed to restore model weights: {e}") from e
        finally:
            if self.track_memory:
                memory_after = self._get_memory_usage()
                self._memory_saved = max(0, self._memory_before - memory_after)
            
            # Clear storage to free memory
            self._deltas.clear()
            self._target_params.clear()
            self._modifications_applied.clear()
            
            # Reset re-entrancy flag
            self._entered = False
    
    def _identify_target_parameters(self):
        """Identify parameters to modify based on layers specification."""
        target_layers = self._get_target_layers()
        
        for layer_name, module in self.model.named_modules():
            if self._should_modify_layer(layer_name, module, target_layers):
                # Add parameters that will be modified
                for param_name, param in module.named_parameters(recurse=False):
                    # Include both trainable and frozen parameters for broader compatibility
                    if param is not None:
                        key = f"{layer_name}.{param_name}"
                        self._target_params[key] = param
    
    def _ensure_contiguity(self):
        """Ensure all target parameters are contiguous to avoid storage aliasing issues."""
        for param_key, param in self._target_params.items():
            if not param.data.is_contiguous():
                # Force contiguity to avoid view/transpose/compile aliasing issues
                param.data = param.data.contiguous()
    
    def _apply_deltas(self):
        """Compute and apply deltas to target parameters."""
        for param_key, param in self._target_params.items():
            # Compute delta based on modification type
            delta = self._compute_delta(param)
            
            # Validate delta shape and compatibility
            if delta.shape != param.shape:
                raise ValueError(f"Delta shape {delta.shape} doesn't match param shape {param.shape} for {param_key}")
            
            # Ensure delta is on same device as parameter
            if delta.device != param.device:
                delta = delta.to(param.device)
            
            # Store delta for restoration (this is our memory-efficient approach)
            self._deltas[param_key] = delta.detach()
            
            # Apply delta in-place
            param.data.add_(delta)
            
            self._modifications_applied.append({
                'param': param_key,
                'type': self.modification_type,
                'delta_norm': delta.norm().item()
            })
    
    def _restore_from_deltas(self):
        """Restore original parameters by subtracting deltas."""
        for param_key, delta in self._deltas.items():
            if param_key in self._target_params:
                param = self._target_params[param_key]
                # Restore by subtracting the delta we applied
                param.data.sub_(delta)
            else:
                raise RuntimeError(f"Parameter {param_key} not found during restoration")
    
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