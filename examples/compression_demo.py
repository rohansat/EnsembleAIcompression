import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from src.adaptive_ndlinear import AdaptiveNdLinear, AdaptiveCompressor

def generate_sample_data(batch_size, input_dim, num_samples):
    """Generate synthetic data for testing."""
    X = torch.randn(num_samples, batch_size, input_dim)
    y = torch.randn(num_samples, batch_size, input_dim // 2)  # Smaller output dimension
    return X, y

def run_comparison():
    # Parameters
    input_dim = 1024
    output_dim = 512
    batch_size = 32
    num_samples = 100
    
    # Generate sample data
    X, y = generate_sample_data(batch_size, input_dim, num_samples)
    
    # Create models
    regular_model = AdaptiveNdLinear((input_dim,), (output_dim,))
    
    # Create compressed model with adaptive compression
    compressor = AdaptiveCompressor(
        target_latency=0.1,
        target_memory=0.5,
        min_compression=0.3,
        max_compression=0.9
    )
    compressed_model = AdaptiveNdLinear((input_dim,), (output_dim,), compressor)
    
    # Metrics storage
    regular_times = []
    compressed_times = []
    regular_memory = []
    compressed_memory = []
    
    # Run comparison
    print("Running comparison...")
    for i in range(num_samples):
        # Regular model
        start_time = time.time()
        _ = regular_model(X[i])
        regular_times.append(time.time() - start_time)
        regular_memory.append(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        
        # Compressed model
        start_time = time.time()
        _ = compressed_model(X[i])
        compressed_times.append(time.time() - start_time)
        compressed_memory.append(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        
    # Calculate statistics
    avg_regular_time = np.mean(regular_times)
    avg_compressed_time = np.mean(compressed_times)
    avg_regular_memory = np.mean(regular_memory)
    avg_compressed_memory = np.mean(compressed_memory)
    
    time_improvement = ((avg_regular_time - avg_compressed_time) / avg_regular_time) * 100
    memory_improvement = ((avg_regular_memory - avg_compressed_memory) / avg_regular_memory) * 100 if avg_regular_memory > 0 else float('nan')
    
    # Prepare detailed results
    results = [
        "\nDetailed Performance Results:",
        "------------------------",
        f"Regular Model:",
        f"  - Average Inference Time: {avg_regular_time:.4f}s",
        f"  - Average Memory Usage: {avg_regular_memory/1e6:.2f}MB",
        f"\nCompressed Model:",
        f"  - Average Inference Time: {avg_compressed_time:.4f}s",
        f"  - Average Memory Usage: {avg_compressed_memory/1e6:.2f}MB",
        f"\nPerformance Improvements:",
        f"  - Time Improvement: {time_improvement:.2f}% faster with compression",
        "  - Memory Usage: " + (
            f"{memory_improvement:.2f}% reduction in memory usage" if not np.isnan(memory_improvement)
            else "The memory comparison shows very small values, which is why we got the 'nan%' improvement (division by near-zero numbers)"
        )
    ]
    
    # Print results to console
    print("\n".join(results))
    
    # Save results to file
    with open('compression_results.txt', 'w') as f:
        f.write("\n".join(results))
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(regular_times, label='Regular')
    plt.plot(compressed_times, label='Compressed')
    plt.title('Inference Time Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Time (s)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot([m/1e6 for m in regular_memory], label='Regular')
    plt.plot([m/1e6 for m in compressed_memory], label='Compressed')
    plt.title('Memory Usage Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Memory (MB)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('compression_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_comparison() 