import torch
from src.adaptive_ndlinear import AdaptiveNdLinear
import os

def main():
    # Create a sample model
    input_dims = (28, 28)  # Example: MNIST image dimensions
    output_dims = (10,)    # Example: 10 classes
    model = AdaptiveNdLinear(input_dims=input_dims, output_dims=output_dims)
    
    # Create a sample input
    batch_size = 32
    x = torch.randn(batch_size, *input_dims)
    
    # Forward pass to generate some metrics
    output = model(x)
    
    # Save the model
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    model.save_model(save_dir, model_name="mnist_classifier")
    
    # Load the model
    loaded_model = AdaptiveNdLinear.load_model(save_dir, model_name="mnist_classifier")
    
    # Verify the loaded model works
    new_output = loaded_model(x)
    
    # Check if outputs match
    assert torch.allclose(output, new_output), "Loaded model produces different outputs!"
    print("Model successfully saved and loaded with matching outputs!")
    
    # Print model details
    print("\nModel Configuration:")
    print(f"Input dimensions: {loaded_model.input_dims}")
    print(f"Output dimensions: {loaded_model.output_dims}")
    print(f"Current compression rate: {loaded_model.current_compression_rate:.2%}")
    print(f"Number of parameters: {sum(p.numel() for p in loaded_model.parameters())}")

if __name__ == "__main__":
    main() 