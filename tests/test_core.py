"""Tests for core SurgicalTheater functionality."""

import torch
import torch.nn as nn
import pytest
from surgical_theater import SurgicalTheater


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.attention = nn.Linear(20, 20)  # This should be auto-detected
        self.linear2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.attention(x)
        return self.linear2(x)


def test_basic_scale_modification():
    """Test basic scaling modification."""
    model = SimpleModel()
    original_weight = model.attention.weight.data.clone()
    
    factor = 0.9
    with SurgicalTheater(model, modification_type="scale", factor=factor):
        # Inside context, weights should be scaled
        expected = original_weight * factor
        assert torch.allclose(model.attention.weight.data, expected, atol=1e-6)
    
    # After context, weights should be restored
    assert torch.allclose(model.attention.weight.data, original_weight, atol=1e-6)


def test_noise_modification():
    """Test noise modification."""
    model = SimpleModel()
    original_weight = model.attention.weight.data.clone()
    
    with SurgicalTheater(model, modification_type="noise", noise_scale=0.01):
        # Inside context, weights should be different (but we can't test exact values due to randomness)
        assert not torch.allclose(model.attention.weight.data, original_weight)
    
    # After context, weights should be restored
    assert torch.allclose(model.attention.weight.data, original_weight, atol=1e-6)


def test_disable_modification():
    """Test disable modification."""
    model = SimpleModel()
    original_weight = model.attention.weight.data.clone()
    
    with SurgicalTheater(model, modification_type="disable"):
        # Inside context, weights should be zero
        assert torch.allclose(model.attention.weight.data, torch.zeros_like(original_weight))
    
    # After context, weights should be restored
    assert torch.allclose(model.attention.weight.data, original_weight, atol=1e-6)


def test_custom_modification():
    """Test custom modification with new API."""
    model = SimpleModel()
    original_weight = model.attention.weight.data.clone()
    
    def custom_delta_fn(param, scale=2.0):
        """Custom function that returns delta directly."""
        return param * (scale - 1.0)  # Return delta, not modified param
    
    with SurgicalTheater(model, modification_type="custom", modification_fn=custom_delta_fn, scale=2.0):
        # Inside context, weights should be doubled
        expected = original_weight * 2.0
        assert torch.allclose(model.attention.weight.data, expected, atol=1e-6)
    
    # After context, weights should be restored
    assert torch.allclose(model.attention.weight.data, original_weight, atol=1e-6)


def test_custom_modification_wrong_shape():
    """Test that custom modification with wrong shape raises error."""
    model = SimpleModel()
    
    def bad_delta_fn(param):
        """Custom function that returns wrong shape."""
        return torch.zeros(5, 5)  # Wrong shape
    
    with pytest.raises(ValueError, match="Delta shape"):
        with SurgicalTheater(model, modification_type="custom", modification_fn=bad_delta_fn):
            pass


def test_re_entrancy_protection():
    """Test that re-entrancy is properly blocked."""
    model = SimpleModel()
    
    theater = SurgicalTheater(model)
    
    with theater:
        # Try to enter the same context again
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with theater:
                pass


def test_memory_tracking():
    """Test memory tracking functionality."""
    model = SimpleModel()
    
    with SurgicalTheater(model, track_memory=True) as theater:
        pass
    
    # Should have some memory statistics
    assert isinstance(theater.memory_saved, float)
    assert isinstance(theater.total_delta_memory_mb, float)


def test_auto_detection_deterministic():
    """Test that auto-detection produces deterministic results."""
    model = SimpleModel()
    
    # Multiple runs should produce identical target layers
    results = []
    for _ in range(3):
        theater = SurgicalTheater(model)
        theater._identify_target_parameters()
        target_keys = sorted(theater._target_params.keys())
        results.append(target_keys)
    
    # All results should be identical
    assert all(result == results[0] for result in results)


def test_contiguity_handling():
    """Test that non-contiguous tensors are handled properly."""
    model = SimpleModel()
    
    # Make a parameter non-contiguous
    original_weight = model.attention.weight.data
    model.attention.weight.data = original_weight.t().t()  # Transpose twice to make non-contiguous
    
    assert not model.attention.weight.data.is_contiguous()
    
    with SurgicalTheater(model, modification_type="scale", factor=0.9):
        # Should work without errors
        pass
    
    # Should be restored correctly
    assert torch.allclose(model.attention.weight.data, original_weight, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])