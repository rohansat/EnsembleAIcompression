import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import time
import psutil
import os
import json

class CompressionMetrics:
    """Tracks performance metrics for adaptive compression."""
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.accuracy_scores = []
        self.compression_rates = []
        
    def update(self, inference_time: float, memory_used: float, 
               accuracy: Optional[float] = None, compression_rate: float = 1.0):
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_used)
        if accuracy is not None:
            self.accuracy_scores.append(accuracy)
        self.compression_rates.append(compression_rate)
        
    def get_average_metrics(self, window_size: int = 100) -> Dict[str, float]:
        """Calculate moving averages of metrics."""
        window = slice(-window_size, None)
        return {
            'avg_inference_time': np.mean(self.inference_times[window]) if self.inference_times else 0.0,
            'avg_memory_usage': np.mean(self.memory_usage[window]) if self.memory_usage else 0.0,
            'avg_compression_rate': np.mean(self.compression_rates[window]) if self.compression_rates else 1.0,
            'avg_accuracy': np.mean(self.accuracy_scores[window]) if self.accuracy_scores else None
        }

class AdaptiveCompressor:
    """Handles dynamic compression of NdLinear layers."""
    def __init__(self, 
                 target_latency: float = 0.1,  # seconds
                 target_memory: float = 0.8,   # percentage of available memory
                 min_compression: float = 0.1,
                 max_compression: float = 0.9):
        self.target_latency = target_latency
        self.target_memory = target_memory
        self.min_compression = min_compression
        self.max_compression = max_compression
        self.metrics = CompressionMetrics()
        
    def get_compression_rate(self, current_metrics: Dict[str, float]) -> float:
        """Determine optimal compression rate based on performance metrics."""
        # Get current system metrics
        current_memory_percent = psutil.virtual_memory().percent / 100.0
        
        # Calculate compression adjustments based on performance
        latency_factor = current_metrics['avg_inference_time'] / self.target_latency
        memory_factor = current_memory_percent / self.target_memory
        
        # Combine factors to determine new compression rate
        compression_rate = current_metrics['avg_compression_rate']
        if latency_factor > 1 or memory_factor > 1:
            # Increase compression if we're exceeding targets
            compression_rate *= 1.1
        elif latency_factor < 0.8 and memory_factor < 0.8:
            # Decrease compression if we have headroom
            compression_rate *= 0.9
            
        # Ensure compression stays within bounds
        return np.clip(compression_rate, self.min_compression, self.max_compression)

class AdaptiveNdLinear(nn.Module):
    """N-dimensional linear layer with dynamic compression capabilities."""
    def __init__(self, 
                 input_dims: Tuple[int, ...],
                 output_dims: Tuple[int, ...],
                 compressor: Optional[AdaptiveCompressor] = None):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.compressor = compressor or AdaptiveCompressor()
        
        # Calculate total input and output sizes
        self.input_size = np.prod(input_dims)
        self.output_size = np.prod(output_dims)
        
        # Initialize weights and bias preserving dimensionality
        self.weight = nn.Parameter(
            torch.randn(self.output_size, *input_dims) / np.sqrt(self.input_size)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_size))
        
        # Register buffer for compression mask
        self.register_buffer('compression_mask', torch.ones_like(self.weight))
        self.current_compression_rate = 1.0
        
    def update_compression_mask(self, compression_rate: float):
        """Update compression mask based on weight magnitudes."""
        if compression_rate == self.current_compression_rate:
            return
            
        with torch.no_grad():  # Don't track gradients for mask updates
            # Calculate number of weights to keep
            total_weights = self.weight.numel()
            num_weights_to_keep = int(total_weights * (1 - compression_rate))
            
            # Calculate weight importance scores
            importance = torch.abs(self.weight.detach())
            threshold = torch.kthvalue(importance.view(-1), 
                                    total_weights - num_weights_to_keep)[0]
            
            # Update mask
            if self.training:
                # Soft mask during training
                self.compression_mask.copy_(
                    torch.sigmoid((importance - threshold) * 10)
                )
            else:
                # Hard mask during inference
                self.compression_mask.copy_(
                    (importance >= threshold).float()
                )
            
            self.current_compression_rate = compression_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start_time = time.time()
        
        # Reshape input while preserving batch dimension
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, -1, *self.input_dims)
        
        # Update compression based on metrics during training
        if self.training:
            metrics = self.compressor.metrics.get_average_metrics()
            new_compression_rate = self.compressor.get_compression_rate(metrics)
            self.update_compression_mask(new_compression_rate)
        
        # Apply compression mask to weights
        effective_weight = self.weight * self.compression_mask
        
        # Perform N-dimensional linear transformation
        # Use einsum for efficient n-dimensional computation
        output = torch.einsum('bi...,oi...->bo', x_reshaped, effective_weight)
        output = output + self.bias
        
        # Reshape output to match output dimensions
        output = output.reshape(batch_size, *self.output_dims)
        
        # Track metrics
        end_time = time.time()
        memory_used = psutil.Process().memory_info().rss / psutil.virtual_memory().total
        self.compressor.metrics.update(
            inference_time=end_time - start_time,
            memory_used=memory_used,
            compression_rate=self.current_compression_rate
        )
        
        return output

    def extra_repr(self) -> str:
        """String representation with layer details."""
        return f'input_dims={self.input_dims}, output_dims={self.output_dims}, compression_rate={self.current_compression_rate:.2f}'

    def save_model(self, save_dir: str, model_name: str = "adaptive_model"):
        """Save the model state and configuration.
        
        Args:
            save_dir: Directory to save the model
            model_name: Name of the model files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        state_dict = {
            'weight': self.weight,
            'bias': self.bias,
            'compression_mask': self.compression_mask,
            'current_compression_rate': self.current_compression_rate
        }
        torch.save(state_dict, model_path)
        
        # Save configuration
        config = {
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'compressor_config': {
                'target_latency': self.compressor.target_latency,
                'target_memory': self.compressor.target_memory,
                'min_compression': self.compressor.min_compression,
                'max_compression': self.compressor.max_compression
            }
        }
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Save metrics history
        metrics = {
            'inference_times': self.compressor.metrics.inference_times,
            'memory_usage': self.compressor.metrics.memory_usage,
            'accuracy_scores': self.compressor.metrics.accuracy_scores,
            'compression_rates': self.compressor.metrics.compression_rates
        }
        metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Model saved successfully in {save_dir}")
        print(f"Files saved: {model_name}.pt, {model_name}_config.json, {model_name}_metrics.json")
    
    @classmethod
    def load_model(cls, save_dir: str, model_name: str = "adaptive_model") -> 'AdaptiveNdLinear':
        """Load a saved model and its configuration.
        
        Args:
            save_dir: Directory containing the saved model
            model_name: Name of the model files
            
        Returns:
            Loaded AdaptiveNdLinear model
        """
        # Load configuration
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Create compressor with saved configuration
        compressor = AdaptiveCompressor(**config['compressor_config'])
        
        # Create model instance
        model = cls(
            input_dims=tuple(config['input_dims']),
            output_dims=tuple(config['output_dims']),
            compressor=compressor
        )
        
        # Load model state
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        state_dict = torch.load(model_path)
        model.weight = nn.Parameter(state_dict['weight'])
        model.bias = nn.Parameter(state_dict['bias'])
        model.compression_mask = state_dict['compression_mask']
        model.current_compression_rate = state_dict['current_compression_rate']
        
        # Load metrics history
        metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                model.compressor.metrics.inference_times = metrics['inference_times']
                model.compressor.metrics.memory_usage = metrics['memory_usage']
                model.compressor.metrics.accuracy_scores = metrics['accuracy_scores']
                model.compressor.metrics.compression_rates = metrics['compression_rates']
        
        print(f"Model loaded successfully from {save_dir}")
        return model 