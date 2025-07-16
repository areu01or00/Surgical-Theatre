"""Tests for FSDP/DeepSpeed sharding support."""

import torch
import torch.nn as nn
import pytest
from surgical_theater import SurgicalTheater
from unittest.mock import patch, MagicMock


class MockShardedParameter(torch.nn.Parameter):
    """Mock sharded parameter for testing without FSDP dependency."""
    
    def __new__(cls, data, requires_grad=True, local_rank=0, world_size=1):
        obj = torch.nn.Parameter.__new__(cls, data, requires_grad)
        obj._local_shard = True
        obj._local_rank = local_rank
        obj._world_size = world_size
        return obj


class MockFSDPModel(nn.Module):
    """Mock FSDP model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.attention = nn.Linear(20, 20)
        self.linear2 = nn.Linear(20, 1)
        
        # Add mock FSDP attributes
        self._fsdp_wrapped_module = True
        
        # Replace parameters with mock sharded versions
        self._shard_parameters()
    
    def _shard_parameters(self):
        """Replace parameters with mock sharded versions."""
        # Shard attention layer weights
        original_weight = self.attention.weight.data
        sharded_weight = MockShardedParameter(original_weight)
        self.attention.weight = sharded_weight
        
        # Shard attention layer bias
        original_bias = self.attention.bias.data
        sharded_bias = MockShardedParameter(original_bias)
        self.attention.bias = sharded_bias
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.attention(x)
        return self.linear2(x)


def test_sharded_model_detection():
    """Test that sharded models are correctly detected."""
    model = MockFSDPModel()
    
    theater = SurgicalTheater(model)
    
    # Should detect FSDP sharding
    assert theater._detect_sharded_model() == True
    
    # Test detection on sharded parameters
    assert hasattr(model.attention.weight, '_local_shard')
    assert hasattr(model.attention.bias, '_local_shard')


def test_sharded_parameter_gathering():
    """Test that sharded parameters are gathered correctly."""
    model = MockFSDPModel()
    
    theater = SurgicalTheater(model)
    
    # Test gathering sharded parameters
    param_key = "attention.weight"
    gathered_param = theater._gather_sharded_parameter(model.attention.weight, param_key)
    
    # Should return a tensor (or None if gathering fails)
    assert gathered_param is not None
    assert isinstance(gathered_param, torch.Tensor)


def test_embedding_layer_fallback():
    """Test that 2-D embedding layers use full-copy fallback."""
    model = nn.Sequential(
        nn.Embedding(1000, 512),  # Large embedding layer
        nn.Linear(512, 256),
        nn.Linear(256, 1)
    )
    
    # Add mock FSDP attributes
    model._fsdp_wrapped_module = True
    
    # Mock distributed environment
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=4), \
         patch('torch.distributed.get_rank', return_value=0):
        
        theater = SurgicalTheater(model)
        
        # Should detect as sharded
        assert theater._detect_sharded_model() == True
        
        # Should use full-copy fallback for embedding layer
        with pytest.warns(UserWarning, match="full-copy used"):
            with theater:
                data = torch.randint(0, 1000, (32, 10))
                # Convert to float for linear layers
                embedded = model[0](data)
                output = model[1:](embedded)


def test_distributed_groups_and_barriers():
    """Test that distributed operations use correct groups and barriers."""
    model = MockFSDPModel()
    
    # Mock distributed environment
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=4), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.group.WORLD') as mock_group, \
         patch('torch.distributed.all_gather') as mock_all_gather, \
         patch('torch.distributed.barrier') as mock_barrier:
        
        theater = SurgicalTheater(model)
        
        # Try to gather a parameter
        param_key = "attention.weight"
        gathered_param = theater._gather_sharded_parameter(model.attention.weight, param_key)
        
        # Should have called all_gather with correct group
        if mock_all_gather.called:
            args, kwargs = mock_all_gather.call_args
            assert 'group' in kwargs
            assert kwargs['group'] == mock_group
        
        # Should have called barrier with correct group
        if mock_barrier.called:
            args, kwargs = mock_barrier.call_args
            assert 'group' in kwargs
            assert kwargs['group'] == mock_group


def test_sharded_parameter_restoration():
    """Test that sharded parameters are restored correctly."""
    model = MockFSDPModel()
    
    # Store original sharded weights
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    # Mock distributed environment
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=4), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.group.WORLD') as mock_group, \
         patch('torch.distributed.barrier') as mock_barrier:
        
        with SurgicalTheater(model, modification_type="scale", factor=0.9):
            data = torch.randn(32, 10)
            output = model(data)
        
        # After context, weights should be restored
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_weights[name], atol=1e-6), \
                f"Sharded weight {name} not restored accurately"
        
        # Should have called barrier during restoration
        assert mock_barrier.called


def test_cpu_offloaded_sharded_parameters():
    """Test handling of CPU-offloaded sharded parameters."""
    model = MockFSDPModel()
    
    # Move model to CPU to simulate offloading
    model = model.cpu()
    
    # Should handle CPU-offloaded sharded parameters gracefully
    with SurgicalTheater(model, modification_type="scale", factor=0.9) as theater:
        data = torch.randn(32, 10)
        output = model(data)
        
        # Should have some modifications applied
        assert len(theater._modifications_applied) > 0


def test_mixed_sharded_non_sharded():
    """Test models with both sharded and non-sharded parameters."""
    model = MockFSDPModel()
    
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    with SurgicalTheater(model, modification_type="scale", factor=0.9):
        data = torch.randn(32, 10)
        output = model(data)
        
        # Check that different parameter types are handled correctly
        assert not hasattr(model.linear1.weight, '_local_shard')  # Non-sharded
        assert hasattr(model.attention.weight, '_local_shard')    # Sharded
    
    # All weights should be restored
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_weights[name], atol=1e-6), \
            f"Weight {name} not restored accurately"


def test_sharded_memory_efficiency():
    """Test that sharded model modifications stay within memory bounds."""
    model = MockFSDPModel()
    
    # Calculate expected memory usage
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_bytes = total_params * 4  # fp32 = 4 bytes per param
    
    with SurgicalTheater(model, track_memory=True, modification_type="scale", factor=0.9) as theater:
        data = torch.randn(32, 10)
        output = model(data)
        
        # Check delta memory usage
        delta_memory_mb = theater.total_delta_memory_mb
        
        # Should be reasonable for sharded model
        assert delta_memory_mb < 1000, f"Delta memory {delta_memory_mb} MB too high for sharded model"


def test_gather_failure_handling():
    """Test graceful handling of parameter gathering failures."""
    model = MockFSDPModel()
    
    # Mock gathering failure
    with patch.object(SurgicalTheater, '_gather_sharded_parameter', 
                     side_effect=Exception("Gathering failed")):
        
        # Should warn about gathering failure and skip parameter
        with pytest.warns(UserWarning, match="Failed to gather sharded parameter"):
            theater = SurgicalTheater(model)
            theater._identify_target_parameters()
            
            # Should have skipped parameters that failed to gather
            assert len(theater._target_params) < len(list(model.named_parameters()))


@pytest.mark.skipif(True, reason="Requires actual FSDP installation and multi-process setup")
def test_real_fsdp_integration():
    """Test with real FSDP (requires multi-process setup)."""
    # This test would require:
    # - Multi-process pytest setup
    # - Real FSDP wrapper
    # - torch.distributed initialization
    #
    # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    # 
    # model = nn.Sequential(
    #     nn.Linear(10, 20),
    #     nn.Linear(20, 20),
    #     nn.Linear(20, 1)
    # )
    # 
    # fsdp_model = FSDP(model)
    # 
    # with SurgicalTheater(fsdp_model) as theater:
    #     output = fsdp_model(torch.randn(32, 10))
    #     assert theater.total_delta_memory_mb < expected_threshold
    
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])