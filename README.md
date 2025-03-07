# Stock-AI: GPU-Optimized Stock Prediction System

This repository contains a GPU-optimized system for training and evaluating machine learning models to predict stock prices for S&P 500 companies.

## Features

- **GPU Optimization**: Utilizes TensorFlow's GPU acceleration with mixed precision training
- **GPU Monitoring**: Real-time monitoring of GPU usage during training
- **Transfer Learning**: Option to use transfer learning for specific stock models
- **Flexible Training**: Train general market models or stock-specific models
- **Performance Metrics**: Comprehensive evaluation of model performance

## Requirements

- Python 3.8+
- TensorFlow 2.16+
- CUDA 12.x and cuDNN (for GPU acceleration)
- pandas, numpy, matplotlib, scikit-learn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-ai.git
cd stock-ai
```

2. Install the required packages:
```bash
pip install tensorflow[and-cuda]==2.16.1 pandas numpy matplotlib scikit-learn
```

## Directory Structure

```
stock-ai/
├── data/
│   ├── processed/       # Processed stock data
│   └── script/          # Data processing scripts
│       ├── general/     # General market data
│       └── specific/    # Stock-specific data
├── models/              # Trained models
├── logs/                # Training logs
├── gpu_optimized_trainer.py  # GPU-optimized trainer class
├── train_sp500_models.py     # Base trainer class
└── colab_train.py       # Script for Google Colab training
```

## Usage

### Training in Google Colab

1. Upload this repository to your Google Drive
2. Open `colab_train.py` in Google Colab
3. Modify the configuration variables as needed:
   - `TRAIN_GENERAL`: Set to True to train the general market model
   - `TRAIN_SPECIFIC`: Set to True to train stock-specific models
   - `EVALUATE`: Set to True to evaluate models
   - `USE_TRANSFER_LEARNING`: Set to True to use transfer learning
   - `SYMBOLS`: List of specific stock symbols to train (e.g., ['AAPL', 'MSFT'])
   - `SAMPLE_FRACTION`: Fraction of data to use for general model training

4. Run the script

### Local Training

```python
from gpu_optimized_trainer import GPUOptimizedTrainer

# Create trainer instance
trainer = GPUOptimizedTrainer(
    processed_dir="data/processed",
    models_dir="models",
    sequence_length=60,
    batch_size=128,
    epochs=50
)

# Train general model
trainer.train_general_model(sample_fraction=1.0)

# Train specific models with transfer learning
trainer.train_all_models(symbols=['AAPL', 'MSFT', 'GOOGL'], use_transfer_learning=True)

# Evaluate models
results = trainer.evaluate_all_models()
print(results)
```

## GPU Optimization Details

The `GPUOptimizedTrainer` class includes several optimizations for GPU training:

1. **Mixed Precision Training**: Uses float16 for faster computation
2. **Memory Growth**: Prevents TensorFlow from allocating all GPU memory at once
3. **XLA Compilation**: Enables XLA JIT compilation for faster execution
4. **tf.data Pipeline**: Optimized data loading with prefetching and caching
5. **GPU Monitoring**: Real-time monitoring of GPU usage during training

## Customization

You can customize the training process by modifying the following parameters:

- **Sequence Length**: Number of days to use for prediction (default: 60)
- **Batch Size**: Number of samples per gradient update (default: 128)
- **Epochs**: Number of training epochs (default: 50)
- **Learning Rate**: Initial learning rate for the optimizer
- **Early Stopping**: Patience for early stopping (default: 10)

## License

[MIT License](LICENSE)

## Acknowledgements

- Data sourced from [Yahoo Finance](https://finance.yahoo.com/)
- Built with [TensorFlow](https://www.tensorflow.org/)
