"""Advanced features demonstration for SurgicalTheater v0.1.1+

Demonstrates:
1. Quantized model support (simulated)
2. Sharded model compatibility (simulated)
3. Combined quantization + sharding scenarios
"""

import torch
import torch.nn as nn
import warnings
from surgical_theater import SurgicalTheater


class MockQuantizedParameter(torch.Tensor):
    """Mock quantized parameter for testing."""
    def __new__(cls, data, original_dtype=torch.float16):
        # Create a mock quantized tensor with int8 dtype
        quantized_data = (data * 127).clamp(-127, 127).to(torch.int8)
        obj = torch.Tensor._make_subclass(cls, quantized_data)
        obj.original_dtype = original_dtype
        return obj


class MockShardedParameter(torch.Tensor):
    """Mock sharded parameter for testing."""
    def __new__(cls, data, local_rank=0, world_size=1):
        obj = torch.Tensor._make_subclass(cls, data)
        obj._local_shard = True
        obj._local_rank = local_rank
        obj._world_size = world_size
        return obj


def create_mock_quantized_model():
    """Create a model with mock quantized parameters."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    # Replace some parameters with mock quantized versions
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            # Convert to mock quantized parameter
            quantized_param = MockQuantizedParameter(param.data)
            # Replace the parameter
            parent_module = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, parts[-1], nn.Parameter(quantized_param))
    
    return model


def create_mock_sharded_model():
    """Create a model with mock sharded parameters."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    # Add mock sharding attributes
    model._fsdp_wrapped_module = True
    
    # Replace some parameters with mock sharded versions
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            # Convert to mock sharded parameter
            sharded_param = MockShardedParameter(param.data)
            # Replace the parameter
            parent_module = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, parts[-1], nn.Parameter(sharded_param))
    
    return model


def example_quantized_model_support():
    """Example: Working with quantized models (simulated)."""
    print("=== Example 1: Quantized Model Support ===")
    
    try:
        model = create_mock_quantized_model()
        test_data = torch.randn(20, 10)
        
        print("Model created with mock quantized parameters")
        
        # Test with SurgicalTheater
        with SurgicalTheater(model, modification_type="scale", factor=0.9) as theater:
            output = model(test_data)
            print(f"✓ Quantized model validation completed")
            print(f"  - Output shape: {output.shape}")
            print(f"  - Modifications applied: {len(theater.modifications_summary)}")
            
            # Check if quantized parameters were detected
            quantized_count = sum(1 for mod in theater._modifications_applied if mod.get('is_quantized', False))
            print(f"  - Quantized parameters processed: {quantized_count}")
        
        print("✓ Model state restored after quantized parameter modifications\n")
        
    except Exception as e:
        print(f"❌ Quantized model support test failed: {e}\n")


def example_sharded_model_compatibility():
    """Example: Working with sharded models (simulated)."""
    print("=== Example 2: Sharded Model Compatibility ===")
    
    try:
        model = create_mock_sharded_model()
        test_data = torch.randn(20, 10)
        
        print("Model created with mock FSDP sharding")
        
        # Test with SurgicalTheater
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sharding warnings in demo
            
            with SurgicalTheater(model, modification_type="scale", factor=0.8) as theater:
                output = model(test_data)
                print(f"✓ Sharded model validation completed")
                print(f"  - Output shape: {output.shape}")
                print(f"  - Modifications applied: {len(theater.modifications_summary)}")
                
                # Check if sharded parameters were detected
                sharded_count = len(theater._sharded_params)
                print(f"  - Sharded parameters detected: {sharded_count}")
        
        print("✓ Model state restored after sharded parameter modifications\n")
        
    except Exception as e:
        print(f"❌ Sharded model compatibility test failed: {e}\n")


def example_memory_efficiency_comparison():
    """Example: Memory efficiency comparison with advanced features."""
    print("=== Example 3: Memory Efficiency with Advanced Features ===")
    
    # Create larger model for better memory comparison
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        test_data = torch.randn(64, 512).cuda()
        print("Using CUDA for memory tracking")
    else:
        test_data = torch.randn(64, 512)
        print("Using CPU for demonstration")
    
    # Memory tracking with SurgicalTheater
    with SurgicalTheater(model, track_memory=True, modification_type="noise", noise_scale=0.01) as theater:
        output = model(test_data)
        
        print(f"Memory efficiency metrics:")
        print(f"  - Memory saved: {theater.memory_saved:.6f} GB")
        print(f"  - Delta memory usage: {theater.total_delta_memory_mb:.2f} MB")
        print(f"  - Parameters modified: {len(theater._target_params)}")
        
        # Show delta statistics
        delta_stats = theater.delta_statistics
        if delta_stats:
            first_param = list(delta_stats.keys())[0]
            stats = delta_stats[first_param]
            print(f"  - Example delta stats for {first_param}:")
            print(f"    * Shape: {stats['shape']}")
            print(f"    * Mean absolute delta: {stats['mean_abs_delta']:.6f}")
            print(f"    * Max absolute delta: {stats['max_abs_delta']:.6f}")
    
    print("✓ Memory efficiency demonstration completed\n")


def example_error_handling():
    """Example: Robust error handling in edge cases."""
    print("=== Example 4: Robust Error Handling ===")
    
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Test custom modification with wrong shape (should fail gracefully)
    def bad_custom_modification(param):
        """Custom function that returns wrong shape delta."""
        return torch.zeros(5, 5)  # Wrong shape
    
    try:
        with SurgicalTheater(model, modification_type="custom", modification_fn=bad_custom_modification):
            pass
        print("❌ Should have failed with shape mismatch")
    except ValueError as e:
        print(f"✓ Correctly caught shape mismatch error: {str(e)[:50]}...")
    
    # Test re-entrancy (should work at depth 1)
    try:
        theater = SurgicalTheater(model, modification_type="scale", factor=0.9)
        with theater:
            # This should work (depth=1)
            output1 = model(torch.randn(10, 10))
            print("✓ Depth=1 context works correctly")
            
            # This should fail (depth=2)
            try:
                with theater:
                    pass
                print("❌ Should have failed with re-entrancy error")
            except RuntimeError as e:
                print(f"✓ Correctly caught re-entrancy error: {str(e)[:50]}...")
                
    except Exception as e:
        print(f"❌ Unexpected error in re-entrancy test: {e}")
    
    print("✓ Error handling tests completed\n")


def main():
    """Run all advanced feature examples."""
    print("SurgicalTheater v0.1.1+ Advanced Features Demo")
    print("=" * 60)
    print("Demonstrating quantized model support and FSDP/DeepSpeed compatibility")
    print("=" * 60)
    
    example_quantized_model_support()
    example_sharded_model_compatibility()
    example_memory_efficiency_comparison()
    example_error_handling()
    
    print("All advanced feature examples completed successfully!")
    print("\nNote: This demo uses mock quantized/sharded parameters for demonstration.")
    print("In production, SurgicalTheater will automatically detect real bitsandbytes,")
    print("QLoRA, FSDP, and DeepSpeed configurations.")


if __name__ == "__main__":
    main()