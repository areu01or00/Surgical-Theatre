"""
SurgicalTheater: Zero-copy context manager for temporary model modifications.

This module provides memory-efficient temporary modifications to PyTorch models,
enabling safe validation and experimentation without memory overhead.
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
    
    Provides zero-copy weight modifications with automatic restoration,
    enabling safe model experimentation with minimal memory overhead.
    
    Examples:
        Basic usage:
        >>> with SurgicalTheater(model) as theater:
        ...     # Model is temporarily modified here
        ...     val_loss = model(validation_data)
        ... # Model automatically restored
        
        With specific modifications:
        >>> with SurgicalTheater(model, layers=[0, 1], modification_type="scale", factor=0.9) as theater:
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
        
        # Storage for original weights
        self._weight_backup = {}
        self._param_backup = {}
        
        # Memory tracking
        self._memory_before = 0
        self._memory_saved = 0
        
        # Modification tracking
        self._modifications_applied = []
    
    def __enter__(self):
        """Enter context and apply temporary modifications."""
        if self.track_memory:
            self._memory_before = self._get_memory_usage()
        
        # Backup weights before modification
        self._backup_weights()
        
        # Apply modifications
        self._apply_modifications()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original weights."""
        # Restore all backed up weights
        self._restore_weights()
        
        if self.track_memory:
            memory_after = self._get_memory_usage()
            self._memory_saved = max(0, self._memory_before - memory_after)
        
        # Clear backups to free memory
        self._weight_backup.clear()
        self._param_backup.clear()
        self._modifications_applied.clear()
    
    def _backup_weights(self):
        """Backup weights that will be modified."""
        target_layers = self._get_target_layers()
        
        for layer_name, module in self.model.named_modules():
            if self._should_backup_layer(layer_name, module, target_layers):
                # Backup parameters
                for param_name, param in module.named_parameters(recurse=False):
                    if param is not None:
                        key = f"{layer_name}.{param_name}"
                        self._param_backup[key] = param.data.clone()
                
                # Backup buffers (batch norm stats, etc.)
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    if buffer is not None:
                        key = f"{layer_name}.{buffer_name}"
                        self._weight_backup[key] = buffer.clone()
    
    def _restore_weights(self):
        """Restore original weights after modification."""
        # Restore parameters
        for key, original_data in self._param_backup.items():
            try:
                *layer_parts, param_name = key.split('.')
                layer_name = '.'.join(layer_parts)
                
                module = self._get_module_by_name(layer_name)
                if hasattr(module, param_name):
                    param = getattr(module, param_name)
                    if param is not None:
                        param.data.copy_(original_data)
            except Exception as e:
                warnings.warn(f"Failed to restore parameter {key}: {e}")
        
        # Restore buffers
        for key, original_data in self._weight_backup.items():
            try:
                *layer_parts, buffer_name = key.split('.')
                layer_name = '.'.join(layer_parts)
                
                module = self._get_module_by_name(layer_name)
                if hasattr(module, buffer_name):
                    buffer = getattr(module, buffer_name)
                    if buffer is not None:
                        buffer.copy_(original_data)
            except Exception as e:
                warnings.warn(f"Failed to restore buffer {key}: {e}")
    
    def _apply_modifications(self):
        """Apply temporary modifications based on modification_type."""
        if self.modification_type == "scale":
            self._apply_scale_modification()
        elif self.modification_type == "noise":
            self._apply_noise_modification()
        elif self.modification_type == "disable":
            self._apply_disable_modification()
        elif self.modification_type == "custom" and self.modification_fn:
            self._apply_custom_modification()
        else:
            warnings.warn(f"Unknown modification type: {self.modification_type}")
    
    def _apply_scale_modification(self):
        """Apply scaling modification to weights."""
        factor = self.kwargs.get('factor', 0.9)
        target_layers = self._get_target_layers()
        
        for layer_name, module in self.model.named_modules():
            if self._should_modify_layer(layer_name, module, target_layers):
                for param_name, param in module.named_parameters(recurse=False):
                    if param is not None and 'weight' in param_name:
                        param.data.mul_(factor)
                        self._modifications_applied.append({
                            'layer': layer_name,
                            'param': param_name,
                            'type': 'scale',
                            'factor': factor
                        })
    
    def _apply_noise_modification(self):
        """Apply noise modification to weights."""
        noise_scale = self.kwargs.get('noise_scale', 0.01)
        target_layers = self._get_target_layers()
        
        for layer_name, module in self.model.named_modules():
            if self._should_modify_layer(layer_name, module, target_layers):
                for param_name, param in module.named_parameters(recurse=False):
                    if param is not None and 'weight' in param_name:
                        noise = torch.randn_like(param.data) * noise_scale
                        param.data.add_(noise)
                        self._modifications_applied.append({
                            'layer': layer_name,
                            'param': param_name,
                            'type': 'noise',
                            'scale': noise_scale
                        })
    
    def _apply_disable_modification(self):
        """Disable layers by zeroing their weights."""
        target_layers = self._get_target_layers()
        
        for layer_name, module in self.model.named_modules():
            if self._should_modify_layer(layer_name, module, target_layers):
                for param_name, param in module.named_parameters(recurse=False):
                    if param is not None and 'weight' in param_name:
                        param.data.zero_()
                        self._modifications_applied.append({
                            'layer': layer_name,
                            'param': param_name,
                            'type': 'disable'
                        })
    
    def _apply_custom_modification(self):
        """Apply custom modification function."""
        if self.modification_fn is None:
            warnings.warn("Custom modification requested but no function provided")
            return
        
        target_layers = self._get_target_layers()
        
        for layer_name, module in self.model.named_modules():
            if self._should_modify_layer(layer_name, module, target_layers):
                # Call custom modification function
                self.modification_fn(module, layer_name, **self.kwargs)
                self._modifications_applied.append({
                    'layer': layer_name,
                    'type': 'custom',
                    'function': self.modification_fn.__name__
                })
    
    def _get_target_layers(self) -> List[str]:
        """Get list of target layer names."""
        if self.layers is None:
            # Auto-detect attention layers
            target_layers = []
            for name, module in self.model.named_modules():
                if any(key in name.lower() for key in ['attention', 'attn', 'self_attn']):
                    target_layers.append(name)
            return target_layers
        
        # Convert layer indices to names
        all_layers = list(self.model.named_modules())
        target_layers = []
        
        for idx in self.layers:
            if 0 <= idx < len(all_layers):
                layer_name, _ = all_layers[idx]
                target_layers.append(layer_name)
        
        return target_layers
    
    def _should_backup_layer(self, layer_name: str, module: nn.Module, target_layers: List[str]) -> bool:
        """Check if layer should be backed up."""
        # Direct match
        if layer_name in target_layers:
            return True
        
        # Parent of target layer
        for target in target_layers:
            if target.startswith(layer_name + '.'):
                return True
        
        return False
    
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
            summary[mod['type']].append(mod['layer'])
        return dict(summary)


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