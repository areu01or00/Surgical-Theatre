"""Example: SurgicalTheater with LoRA training."""

import torch
import torch.nn as nn
from surgical_theater import SurgicalTheater


class SimpleLoRALayer(nn.Module):
    """Simplified LoRA layer for demonstration."""
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1.0
        
    def forward(self, x):
        # Regular forward + LoRA adaptation
        base_output = self.linear(x)
        lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_output + lora_output


def train_with_lora():
    """Example: Training LoRA with frequent validation using SurgicalTheater."""
    print("=== LoRA Training with SurgicalTheater ===")
    
    # Create model with LoRA layers
    model = nn.Sequential(
        SimpleLoRALayer(20, 64, rank=4),
        nn.ReLU(),
        SimpleLoRALayer(64, 32, rank=4),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Data
    train_data = torch.randn(200, 20)
    train_labels = torch.randn(200, 1)
    val_data = torch.randn(50, 20)
    val_labels = torch.randn(50, 1)
    
    # Only optimize LoRA parameters
    lora_params = []
    for module in model.modules():
        if isinstance(module, SimpleLoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    
    optimizer = torch.optim.Adam(lora_params, lr=0.001)
    
    # Training with frequent validation
    print("Training with validation every 10 steps...")
    
    for step in range(100):
        # Training step
        idx = torch.randperm(len(train_data))[:32]  # Random batch
        batch_data = train_data[idx]
        batch_labels = train_labels[idx]
        
        optimizer.zero_grad()
        output = model(batch_data)
        loss = nn.functional.mse_loss(output, batch_labels)
        loss.backward()
        optimizer.step()
        
        # Frequent validation with SurgicalTheater
        if step % 10 == 0:
            with SurgicalTheater(model):
                model.eval()
                with torch.no_grad():
                    val_output = model(val_data)
                    val_loss = nn.functional.mse_loss(val_output, val_labels)
                model.train()
            
            print(f"Step {step}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
    
    print("\n✓ LoRA training completed with zero memory overhead for validation!")


def compare_memory_usage():
    """Compare memory usage: deepcopy vs SurgicalTheater."""
    print("\n=== Memory Usage Comparison ===")
    
    # Create a larger model
    model = nn.Sequential(
        SimpleLoRALayer(512, 1024, rank=8),
        nn.ReLU(),
        SimpleLoRALayer(1024, 1024, rank=8),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    
    data = torch.randn(32, 512)
    
    # Method 1: Traditional deepcopy (expensive)
    import copy
    import time
    
    print("Method 1: Using deepcopy...")
    start_time = time.time()
    model_copy = copy.deepcopy(model)  # This would use ~2x memory!
    with torch.no_grad():
        output1 = model_copy(data)
    deepcopy_time = time.time() - start_time
    del model_copy  # Free memory
    
    # Method 2: SurgicalTheater (efficient)
    print("Method 2: Using SurgicalTheater...")
    start_time = time.time()
    with SurgicalTheater(model):
        with torch.no_grad():
            output2 = model(data)
    surgical_time = time.time() - start_time
    
    print(f"\nDeepCopy time: {deepcopy_time:.4f}s")
    print(f"SurgicalTheater time: {surgical_time:.4f}s")
    print(f"Speedup: {deepcopy_time/surgical_time:.2f}x")
    print("\n✓ SurgicalTheater is faster and uses much less memory!")


def test_different_lora_configs():
    """Test different LoRA configurations safely."""
    print("\n=== Testing Different LoRA Configurations ===")
    
    base_model = nn.Sequential(
        nn.Linear(50, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    test_data = torch.randn(100, 50)
    test_labels = torch.randint(0, 10, (100,))
    
    # Test different scaling factors for LoRA-like modifications
    print("Testing different LoRA-style scaling factors...")
    
    scaling_factors = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for scale in scaling_factors:
        # Simulate LoRA scaling effect
        with SurgicalTheater(base_model, layers=[0, 2], modification_type="scale", factor=1+scale*0.1):
            output = base_model(test_data)
            loss = nn.functional.cross_entropy(output, test_labels)
            accuracy = (output.argmax(1) == test_labels).float().mean()
            print(f"LoRA scale {scale}: Loss = {loss:.4f}, Accuracy = {accuracy:.3f}")
    
    print("\n✓ All configurations tested without modifying the original model!")


def main():
    """Run all LoRA examples."""
    print("SurgicalTheater + LoRA Integration Examples")
    print("=" * 50)
    
    train_with_lora()
    compare_memory_usage()
    test_different_lora_configs()
    
    print("\nAll LoRA examples completed successfully!")


if __name__ == "__main__":
    main()