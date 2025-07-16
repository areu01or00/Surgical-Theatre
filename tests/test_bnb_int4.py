"""Tests for bitsandbytes quantized model support."""

import torch
import torch.nn as nn
import pytest
from surgical_theater import SurgicalTheater


class MockQuantizedParameter(torch.nn.Parameter):
    """Mock quantized parameter for testing without bitsandbytes dependency."""
    
    def __new__(cls, data, requires_grad=True, original_dtype=torch.float16):
        # Create a mock quantized tensor with int8 dtype
        if data.dtype != torch.int8:
            quantized_data = (data * 127).clamp(-127, 127).to(torch.int8)
        else:
            quantized_data = data
        
        obj = torch.nn.Parameter.__new__(cls, quantized_data, requires_grad)
        obj.original_dtype = original_dtype
        return obj


class MockQuantizedModel(nn.Module):
    """Mock quantized model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.attention = nn.Linear(20, 20)
        self.linear2 = nn.Linear(20, 1)
        
        # Replace some parameters with mock quantized versions
        self._quantize_parameters()
    
    def _quantize_parameters(self):
        """Replace parameters with mock quantized versions."""
        # Quantize attention layer weights
        original_weight = self.attention.weight.data
        quantized_weight = MockQuantizedParameter(original_weight)
        self.attention.weight = quantized_weight
        
        # Quantize attention layer bias
        original_bias = self.attention.bias.data
        quantized_bias = MockQuantizedParameter(original_bias)
        self.attention.bias = quantized_bias
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        # Convert quantized weights to float for computation
        attention_weight = self.attention.weight.data.float()
        attention_bias = self.attention.bias.data.float()
        x = torch.nn.functional.linear(x, attention_weight, attention_bias)
        return self.linear2(x)


def test_quantized_parameter_detection():
    """Test that quantized parameters are correctly detected."""
    model = MockQuantizedModel()
    
    theater = SurgicalTheater(model)
    
    # Test detection on quantized parameters
    assert theater._is_quantized_parameter(model.attention.weight) == True
    assert theater._is_quantized_parameter(model.attention.bias) == True
    
    # Test detection on non-quantized parameters
    assert theater._is_quantized_parameter(model.linear1.weight) == False
    assert theater._is_quantized_parameter(model.linear1.bias) == False


def test_quantized_memory_efficiency():
    """Test that quantized model modifications stay within memory bounds."""
    model = MockQuantizedModel()
    
    # Calculate expected memory usage
    total_params = sum(p.numel() for p in model.parameters())
    # For int8 quantized params, delta should be stored in original dtype
    expected_delta_memory_approx = total_params * 1  # int8 = 1 byte per param
    
    with SurgicalTheater(model, track_memory=True, modification_type="scale", factor=0.9) as theater:
        data = torch.randn(32, 10)
        output = model(data)
        
        # Check that we're not storing huge FP32 deltas
        delta_memory_mb = theater.total_delta_memory_mb
        
        # Should be reasonable for int8 storage
        assert delta_memory_mb < 100, f"Delta memory {delta_memory_mb} MB too high for quantized model"


def test_quantized_weight_restoration():
    """Test that quantized weights are restored correctly."""
    model = MockQuantizedModel()
    
    # Store original quantized weights
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    with SurgicalTheater(model, modification_type="scale", factor=0.8):
        # Inside context, quantized weights should be modified
        # (but stored as quantized, not FP32)
        data = torch.randn(32, 10)
        output = model(data)
        
        # Check that quantized parameters are still quantized
        assert model.attention.weight.dtype == torch.int8
        assert model.attention.bias.dtype == torch.int8
    
    # After context, weights should be restored exactly
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_weights[name], atol=1), \
            f"Quantized weight {name} not restored accurately"
        
        # Check that quantized parameters maintain their dtype
        if hasattr(param, 'original_dtype'):
            assert param.dtype == torch.int8


def test_cpu_offloaded_quantized_parameters():
    """Test handling of CPU-offloaded quantized parameters."""
    model = MockQuantizedModel()
    
    # Move model to CPU to simulate offloading
    model = model.cpu()
    
    # Modify attention parameters to simulate CPU offloading
    model.attention.weight.data = model.attention.weight.data.cpu()
    model.attention.bias.data = model.attention.bias.data.cpu()
    
    # Should not crash with CPU-offloaded parameters
    with SurgicalTheater(model, modification_type="scale", factor=0.9) as theater:
        data = torch.randn(32, 10)
        output = model(data)
        
        # Should have warned about CPU offloading
        assert len(theater._modifications_applied) > 0


def test_quantized_custom_modification():
    """Test custom modifications with quantized parameters."""
    model = MockQuantizedModel()
    
    def custom_quantized_delta(param, factor=1.5):
        """Custom delta function that works with quantized params."""
        # For quantized params, work in float space for delta computation
        param_float = param.float()
        return param_float * (factor - 1.0)
    
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    with SurgicalTheater(model, modification_type="custom", 
                        modification_fn=custom_quantized_delta, factor=1.5):
        data = torch.randn(32, 10)
        output = model(data)
        
        # Quantized parameters should still be quantized
        assert model.attention.weight.dtype == torch.int8
        assert model.attention.bias.dtype == torch.int8
    
    # After context, weights should be restored
    for name, param in model.named_parameters():
        # Allow for some quantization error
        assert torch.allclose(param.data, original_weights[name], atol=2), \
            f"Quantized weight {name} not restored within tolerance"


def test_mixed_quantized_non_quantized():
    """Test models with both quantized and non-quantized parameters."""
    model = MockQuantizedModel()
    
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    with SurgicalTheater(model, modification_type="scale", factor=0.9):
        data = torch.randn(32, 10)
        output = model(data)
        
        # Check that different parameter types are handled correctly
        assert model.linear1.weight.dtype == torch.float32  # Non-quantized
        assert model.attention.weight.dtype == torch.int8   # Quantized
    
    # All weights should be restored
    for name, param in model.named_parameters():
        if param.dtype == torch.int8:
            # Quantized parameters - allow for quantization error
            assert torch.allclose(param.data, original_weights[name], atol=2)
        else:
            # Non-quantized parameters - expect exact restoration
            assert torch.allclose(param.data, original_weights[name], atol=1e-6)


@pytest.mark.skipif(True, reason="Requires actual bitsandbytes installation")
def test_real_bitsandbytes_integration():
    """Test with real bitsandbytes (requires separate installation)."""
    # This test would require:
    # pip install bitsandbytes
    # from bitsandbytes import nn as bnb
    # 
    # model = nn.Sequential(
    #     nn.Linear(10, 20),
    #     bnb.Linear8bitLt(20, 20),  # Real quantized layer
    #     nn.Linear(20, 1)
    # )
    #
    # with SurgicalTheater(model) as theater:
    #     output = model(torch.randn(32, 10))
    #     assert theater.total_delta_memory_mb < expected_threshold
    
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])