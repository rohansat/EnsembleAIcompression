import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from adaptive_ndlinear import AdaptiveNdLinear, AdaptiveCompressor

class AdaptiveImageClassifier(nn.Module):
    """Example model using AdaptiveNdLinear for image classification."""
    def __init__(self, image_size=(28, 28), num_classes=10):
        super().__init__()
        
        # Create a compressor with custom settings
        self.compressor = AdaptiveCompressor(
            target_latency=0.005,  # 5ms target latency
            target_memory=0.7,     # Use up to 70% memory
            min_compression=0.2,    # Minimum 20% compression
            max_compression=0.8     # Maximum 80% compression
        )
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate dimensions after convolutions and pooling
        conv_output_size = (image_size[0] // 4, image_size[1] // 4)
        
        # Adaptive NdLinear layer instead of traditional flatten + linear
        self.adaptive_linear = AdaptiveNdLinear(
            input_dims=(64, *conv_output_size),  # Preserve channel and spatial dimensions
            output_dims=(num_classes,),          # Output classes
            compressor=self.compressor
        )
        
    def forward(self, x):
        # Convolutional feature extraction
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Use adaptive NdLinear layer
        # No need to flatten - it preserves the dimensional structure
        x = self.adaptive_linear(x)
        x = x.squeeze()  # Remove any extra dimensions
        return x  # Return raw logits, let loss function handle the log_softmax

def train_model(epochs=5, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Train the adaptive model on MNIST."""
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model and move to device
    model = AdaptiveImageClassifier().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss instead of NLLLoss
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)  # CrossEntropyLoss handles the log_softmax
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                # Print compression metrics
                metrics = model.adaptive_linear.compressor.metrics.get_average_metrics()
                print(f'Current compression rate: {model.adaptive_linear.current_compression_rate:.2f}')
                print(f'Average inference time: {metrics["avg_inference_time"]*1000:.2f}ms')
                print(f'Memory usage: {metrics["avg_memory_usage"]*100:.1f}%\n')
        
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        
        # Update accuracy in metrics
        model.adaptive_linear.compressor.metrics.update(
            inference_time=metrics["avg_inference_time"],
            memory_used=metrics["avg_memory_usage"],
            accuracy=accuracy/100,
            compression_rate=model.adaptive_linear.current_compression_rate
        )

if __name__ == "__main__":
    train_model() 