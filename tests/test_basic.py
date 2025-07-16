"""Basic tests for SurgicalTheater functionality."""

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


def test_memory_usage_within_readme_bounds():
    """Test that memory usage stays within README promises (±5%)."""
    model = SimpleModel()
    
    # Calculate expected memory usage
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_bytes = total_params * 4  # fp32 = 4 bytes per param
    
    # Expected: ~1 parameter set extra memory
    expected_delta_memory = param_memory_bytes
    tolerance = 0.05  # ±5%
    
    with SurgicalTheater(model, track_memory=True) as theater:
        # Do some operation
        data = torch.randn(32, 10)
        output = model(data)
        
        # Check delta memory usage
        actual_delta_memory = theater.total_delta_memory_mb * 1024 * 1024  # Convert to bytes
        
        # Should be within ±5% of expected
        assert abs(actual_delta_memory - expected_delta_memory) / expected_delta_memory < tolerance, \
            f"Memory usage {actual_delta_memory} bytes not within ±5% of expected {expected_delta_memory} bytes"


def test_requires_grad_preservation():
    """Test that requires_grad flags are preserved after context exit."""
    model = SimpleModel()
    
    # Set different requires_grad flags
    model.linear1.weight.requires_grad = True
    model.linear1.bias.requires_grad = False
    model.attention.weight.requires_grad = True
    model.attention.bias.requires_grad = False
    
    # Store original flags
    original_flags = {
        'linear1.weight': model.linear1.weight.requires_grad,
        'linear1.bias': model.linear1.bias.requires_grad,
        'attention.weight': model.attention.weight.requires_grad,
        'attention.bias': model.attention.bias.requires_grad,
    }
    
    with SurgicalTheater(model, modification_type="scale", factor=0.9):
        # Inside context, flags might be modified
        pass
    
    # After context, flags should be restored
    assert model.linear1.weight.requires_grad == original_flags['linear1.weight']
    assert model.linear1.bias.requires_grad == original_flags['linear1.bias']
    assert model.attention.weight.requires_grad == original_flags['attention.weight']
    assert model.attention.bias.requires_grad == original_flags['attention.bias']


def test_weight_restoration_accuracy():
    """Test that weights are restored exactly after context exit."""
    model = SimpleModel()
    
    # Store original weights
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    with SurgicalTheater(model, modification_type="scale", factor=0.5):
        # Inside context, weights should be modified
        for name, param in model.named_parameters():
            if 'attention' in name:  # Only attention layers are modified by default
                assert not torch.allclose(param.data, original_weights[name])
    
    # After context, weights should be restored exactly
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_weights[name], atol=1e-6), \
            f"Weight {name} not restored accurately"


def test_gradient_flow_preservation():
    """Test that gradient flow is preserved after context exit."""
    model = SimpleModel()
    data = torch.randn(32, 10)
    target = torch.randn(32, 1)
    
    # Training step before context
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Store gradients
    original_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            original_grads[name] = param.grad.clone()
    
    optimizer.step()
    
    # Use SurgicalTheater for validation
    with SurgicalTheater(model, modification_type="scale", factor=0.9):
        val_output = model(data)
        val_loss = nn.functional.mse_loss(val_output, target)
    
    # Training step after context
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Gradients should still flow correctly
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name} after context"
        assert param.requires_grad, f"requires_grad=False for {name} after context"


def test_nested_contexts_depth_1():
    """Test that depth=1 nested contexts work correctly."""
    model = SimpleModel()
    
    theater = SurgicalTheater(model, modification_type="scale", factor=0.9)
    
    with theater:
        # This should work (depth=1)
        data = torch.randn(10, 10)
        output1 = model(data)
        
        # This should fail (depth=2)
        with pytest.raises(RuntimeError, match="Nested SurgicalTheater"):
            with theater:
                pass


def test_exception_safety():
    """Test that model is restored even if exception occurs."""
    model = SimpleModel()
    
    # Store original weights
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    try:
        with SurgicalTheater(model, modification_type="scale", factor=0.5):
            # Cause an exception
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Even with exception, weights should be restored
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_weights[name], atol=1e-6), \
            f"Weight {name} not restored after exception"


def test_custom_modification_delta_api():
    """Test that custom modification functions work with delta API."""
    model = SimpleModel()
    
    def custom_delta_fn(param, scale=2.0):
        """Custom function that returns delta directly."""
        return param * (scale - 1.0)  # Return delta, not modified param
    
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    with SurgicalTheater(model, modification_type="custom", 
                        modification_fn=custom_delta_fn, scale=2.0):
        # Inside context, attention weights should be doubled
        for name, param in model.named_parameters():
            if 'attention' in name:
                expected = original_weights[name] * 2.0
                assert torch.allclose(param.data, expected, atol=1e-6)
    
    # After context, weights should be restored
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_weights[name], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])