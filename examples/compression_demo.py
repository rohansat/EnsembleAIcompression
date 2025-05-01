import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from src.adaptive_ndlinear import AdaptiveNdLinear, AdaptiveCompressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def generate_sample_data(batch_size, input_dim, num_samples):
    """Generate test data for benchmarking compression performance"""
    X = torch.randn(num_samples, batch_size, input_dim)
    # Generate target data with a known pattern for accuracy comparison
    y = torch.sin(X.mean(dim=-1, keepdim=True)) + torch.randn(num_samples, batch_size, 1) * 0.1
    return X, y

def calculate_metrics(predictions, targets):
    """Calculate accuracy and RMSE metrics"""
    # Convert to numpy for easier calculation
    pred_np = predictions.detach().numpy()
    target_np = targets.detach().numpy()
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(target_np.flatten(), pred_np.flatten()))
    
    # Calculate accuracy (within 10% of target)
    tolerance = np.abs(target_np) * 0.1
    accurate = np.abs(pred_np - target_np) <= tolerance
    accuracy = np.mean(accurate) * 100
    
    return accuracy, rmse

def run_comparison():
    # Model configuration
    input_dim = 1024
    output_dim = 1  # Single output for easier accuracy comparison
    batch_size = 32
    num_samples = 100
    
    # Load test data
    X, y = generate_sample_data(batch_size, input_dim, num_samples)
    
    # Initialize models
    regular_model = AdaptiveNdLinear((input_dim,), (output_dim,))
    
    # TODO: Experiment with different compression settings
    compressor = AdaptiveCompressor(
        target_latency=0.1,    # 100ms target
        target_memory=0.5,     # 50% memory target
        min_compression=0.3,   # Minimum 30% compression
        max_compression=0.9    # Maximum 90% compression
    )
    compressed_model = AdaptiveNdLinear((input_dim,), (output_dim,), compressor)
    
    # Performance tracking
    regular_latencies = []
    compressed_latencies = []
    regular_memory = []
    compressed_memory = []
    regular_accuracies = []
    compressed_accuracies = []
    regular_rmses = []
    compressed_rmses = []
    
    print("Running comparison with larger model dimensions...")
    for i in range(num_samples):
        # Measure regular model performance
        _ = regular_model(X[i])  # Warm-up run
        start_time = time.time()
        regular_output = regular_model(X[i])
        regular_latencies.append(time.time() - start_time)
        regular_memory.append(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        
        # Calculate regular model accuracy and RMSE
        acc, rmse = calculate_metrics(regular_output, y[i])
        regular_accuracies.append(acc)
        regular_rmses.append(rmse)
        
        # Measure compressed model performance
        _ = compressed_model(X[i])  # Warm-up run
        start_time = time.time()
        compressed_output = compressed_model(X[i])
        compressed_latencies.append(time.time() - start_time)
        compressed_memory.append(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        
        # Calculate compressed model accuracy and RMSE
        acc, rmse = calculate_metrics(compressed_output, y[i])
        compressed_accuracies.append(acc)
        compressed_rmses.append(rmse)
        
    # Calculate metrics
    avg_regular_latency = np.mean(regular_latencies)
    avg_compressed_latency = np.mean(compressed_latencies)
    avg_regular_memory = np.mean(regular_memory)
    avg_compressed_memory = np.mean(compressed_memory)
    avg_regular_accuracy = np.mean(regular_accuracies)
    avg_compressed_accuracy = np.mean(compressed_accuracies)
    avg_regular_rmse = np.mean(regular_rmses)
    avg_compressed_rmse = np.mean(compressed_rmses)
    
    latency_improvement = ((avg_regular_latency - avg_compressed_latency) / avg_regular_latency) * 100
    memory_improvement = ((avg_regular_memory - avg_compressed_memory) / avg_regular_memory) * 100 if avg_regular_memory > 0 else float('nan')
    accuracy_difference = avg_compressed_accuracy - avg_regular_accuracy
    rmse_difference = avg_compressed_rmse - avg_regular_rmse
    
    # Format results
    results = [
        "\nDetailed Performance Results:",
        "------------------------",
        f"Regular Model:",
        f"  - Average Latency: {avg_regular_latency:.4f}s",
        f"  - Average Memory Usage: {avg_regular_memory/1e6:.2f}MB",
        f"  - Average Accuracy: {avg_regular_accuracy:.2f}%",
        f"  - Average RMSE: {avg_regular_rmse:.4f}",
        f"\nCompressed Model:",
        f"  - Average Latency: {avg_compressed_latency:.4f}s",
        f"  - Average Memory Usage: {avg_compressed_memory/1e6:.2f}MB",
        f"  - Average Accuracy: {avg_compressed_accuracy:.2f}%",
        f"  - Average RMSE: {avg_compressed_rmse:.4f}",
        f"\nPerformance Improvements:",
        f"  - Latency Improvement: {latency_improvement:.2f}% faster with compression",
        f"  - Accuracy Impact: {accuracy_difference:+.2f}% with compression",
        f"  - RMSE Change: {rmse_difference:+.4f} with compression",
        "  - Memory Usage: " + (
            f"{memory_improvement:.2f}% reduction in memory usage" if not np.isnan(memory_improvement)
            else "The memory comparison shows very small values, which is why we got the 'nan%' improvement (division by near-zero numbers)"
        )
    ]
    
    print("\n".join(results))
    
    # Save benchmark results
    with open('compression_results.txt', 'w') as f:
        f.write("\n".join(results))
    
    # Generate comparison plots
    plt.figure(figsize=(15, 5))
    
    # Latency plot
    plt.subplot(1, 3, 1)
    plt.plot(regular_latencies, label='Regular')
    plt.plot(compressed_latencies, label='Compressed')
    plt.title('Latency Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Latency (s)')
    plt.legend()
    
    # Memory plot
    plt.subplot(1, 3, 2)
    plt.plot([m/1e6 for m in regular_memory], label='Regular')
    plt.plot([m/1e6 for m in compressed_memory], label='Compressed')
    plt.title('Memory Usage Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Memory (MB)')
    plt.legend()
    
    # Accuracy/RMSE plot
    plt.subplot(1, 3, 3)
    plt.plot(regular_accuracies, label='Regular Accuracy')
    plt.plot(compressed_accuracies, label='Compressed Accuracy')
    plt.title('Accuracy Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('compression_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_comparison() 