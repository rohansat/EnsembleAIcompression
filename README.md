# Adaptive Compression with NdLinear

This project implements an adaptive compression system for neural networks using NdLinear layers. It automatically adjusts compression rates based on system performance metrics and resource availability.

## Key Features

- **Dynamic Compression**: Automatically adjusts compression rates based on:
  - Inference latency
  - Memory usage
  - Model accuracy
  - System resources

- **Dimensional Preservation**: Uses NdLinear to maintain multi-dimensional data structure without flattening
- **Resource-Aware**: Monitors system metrics to optimize performance
- **Adaptive Control**: Intelligent compression adjustment based on real-time feedback

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd adaptive_compression_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main component is the `AdaptiveNdLinear` layer, which can be used as a drop-in replacement for `nn.Linear` while preserving dimensional structure:

```python
from adaptive_ndlinear import AdaptiveNdLinear

# Create an adaptive layer
layer = AdaptiveNdLinear(
    input_dims=(64, 28, 28),  # Input dimensions (channels, height, width)
    output_dims=(10,)         # Output dimensions (num_classes)
)
```

### Example

Check out `examples/image_classifier.py` for a complete example using MNIST:

```bash
python examples/image_classifier.py
```

This example demonstrates:
- Using AdaptiveNdLinear in a CNN
- Dynamic compression during training
- Performance monitoring and adaptation
- Accuracy preservation with reduced resource usage

## Architecture

The system consists of three main components:

1. **AdaptiveNdLinear**: The core layer that preserves dimensional structure while enabling compression
2. **CompressionMetrics**: Tracks performance metrics and resource usage
3. **AdaptiveCompressor**: Controls compression rates based on monitored metrics

## Performance Metrics

The system tracks:
- Inference time per batch
- Memory usage
- Model accuracy
- Compression rates

These metrics are used to dynamically adjust compression levels, ensuring optimal performance within resource constraints.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 