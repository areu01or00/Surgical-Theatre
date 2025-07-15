"""Basic usage examples for SurgicalTheater."""

import torch
import torch.nn as nn
from surgical_theater import SurgicalTheater, surgical_theater


def example_simple_validation():
    """Example: Basic validation without memory overhead."""
    print("=== Example 1: Simple Validation ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Training data
    train_data = torch.randn(100, 10)
    train_labels = torch.randn(100, 1)
    
    # Validation data
    val_data = torch.randn(20, 10)
    val_labels = torch.randn(20, 1)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop with safe validation
    for epoch in range(5):
        # Training step
        optimizer.zero_grad()
        train_output = model(train_data)
        train_loss = nn.functional.mse_loss(train_output, train_labels)
        train_loss.backward()
        optimizer.step()
        
        # Safe validation with SurgicalTheater
        with SurgicalTheater(model):
            model.eval()
            with torch.no_grad():
                val_output = model(val_data)
                val_loss = nn.functional.mse_loss(val_output, val_labels)
            model.train()
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print("✓ Model state preserved after validation\n")


def example_hyperparameter_testing():
    """Example: Test different configurations without memory overhead."""
    print("=== Example 2: Hyperparameter Testing ===")
    
    # Create a model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    test_data = torch.randn(50, 10)
    test_labels = torch.randn(50, 1)
    
    # Test different scaling factors
    scaling_factors = [0.5, 0.8, 0.9, 1.1, 1.2]
    
    for factor in scaling_factors:
        with surgical_theater(model, modification_type="scale", factor=factor):
            output = model(test_data)
            loss = nn.functional.mse_loss(output, test_labels)
            print(f"Scale factor {factor}: Loss = {loss:.4f}")
    
    # Original model is unchanged
    print("✓ Original model preserved after all tests\n")


def example_layer_ablation():
    """Example: Layer ablation study."""
    print("=== Example 3: Layer Ablation Study ===")
    
    # Create a deeper model
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(10, 32),
                nn.Linear(32, 32),
                nn.Linear(32, 32),
                nn.Linear(32, 1)
            ])
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                if x.shape[-1] != 1:  # Don't apply ReLU to final layer
                    x = torch.relu(x)
            return x
    
    model = SimpleTransformer()
    test_data = torch.randn(50, 10)
    
    # Test disabling each layer
    for i in range(len(model.layers) - 1):  # Don't disable output layer
        with SurgicalTheater(model, layers=[i], modification_type="disable"):
            output = model(test_data)
            print(f"Layer {i} disabled: Output mean = {output.mean():.4f}")
    
    # Test with all layers active
    output = model(test_data)
    print(f"All layers active: Output mean = {output.mean():.4f}")
    print("✓ Model restored after ablation study\n")


def example_noise_robustness():
    """Example: Test model robustness to weight perturbations."""
    print("=== Example 4: Noise Robustness Testing ===")
    
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    test_data = torch.randn(100, 10)
    
    # Get baseline performance
    with torch.no_grad():
        baseline_output = model(test_data)
        baseline_mean = baseline_output.mean().item()
    
    # Test different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    
    for noise in noise_levels:
        with surgical_theater(model, modification_type="noise", noise_scale=noise):
            with torch.no_grad():
                noisy_output = model(test_data)
                noisy_mean = noisy_output.mean().item()
                diff = abs(noisy_mean - baseline_mean)
                print(f"Noise level {noise}: Output diff = {diff:.4f}")
    
    print("✓ Model weights restored after noise testing\n")


def example_memory_tracking():
    """Example: Track memory usage with SurgicalTheater."""
    print("=== Example 5: Memory Usage Tracking ===")
    
    # Create a larger model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        data = torch.randn(32, 512).cuda()
    else:
        data = torch.randn(32, 512)
    
    # Track memory with SurgicalTheater
    with SurgicalTheater(model, track_memory=True) as theater:
        output = model(data)
        print(f"Memory saved: {theater.memory_saved:.4f} GB")
        print(f"Modifications applied: {theater.modifications_summary}")
    
    print("✓ Memory efficiently managed\n")


def example_custom_modification():
    """Example: Custom modification function."""
    print("=== Example 6: Custom Modifications ===")
    
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    # Define custom modification function
    def attention_style_modification(module, layer_name, temperature=1.0):
        """Apply attention-style scaling to weights."""
        if hasattr(module, 'weight'):
            # Apply softmax-like scaling
            weight = module.weight.data
            scaled = torch.softmax(weight / temperature, dim=1)
            module.weight.data = scaled * weight.shape[1]  # Rescale
    
    test_data = torch.randn(20, 10)
    
    # Test different temperatures
    temperatures = [0.5, 1.0, 2.0]
    
    for temp in temperatures:
        with SurgicalTheater(
            model, 
            modification_type="custom",
            modification_fn=attention_style_modification,
            temperature=temp
        ):
            output = model(test_data)
            print(f"Temperature {temp}: Output std = {output.std():.4f}")
    
    print("✓ Custom modifications applied and reverted\n")


def main():
    """Run all examples."""
    print("SurgicalTheater Usage Examples")
    print("=" * 50)
    
    example_simple_validation()
    example_hyperparameter_testing()
    example_layer_ablation()
    example_noise_robustness()
    example_memory_tracking()
    example_custom_modification()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()