# Stock-AI: Stock Price Prediction System

A comprehensive system for predicting stock prices using deep learning models (TensorFlow and PyTorch LSTM models), with message queuing for distributed processing via RabbitMQ and Celery.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Data Collection and Processing](#data-collection-and-processing)
5. [Model Training](#model-training)
6. [Model Comparison](#model-comparison)
7. [Prediction Service](#prediction-service)
8. [Message Queue System](#message-queue-system)
9. [Task Processing](#task-processing)
10. [Monitoring Tools](#monitoring-tools)
11. [Usage](#usage)
12. [Troubleshooting](#troubleshooting)

## Overview

This project implements a complete stock price prediction system that:

- Collects and processes historical stock data (Google/Alphabet)
- Trains both TensorFlow and PyTorch LSTM models
- Compares model performance with metrics (MAE, RMSE, R², MAPE)
- Makes predictions for future stock prices
- Distributes predictions through a message queue for parallel processing
- Implements monitoring and management tools

The system uses a microservice architecture with loosely coupled components that communicate via RabbitMQ message queues.

## System Architecture

The system consists of the following components:

1. **Data Collection & Processing Pipeline**: Fetches and preprocesses stock data
2. **Model Training System**: Trains and evaluates LSTM models
3. **Prediction Service**: Makes predictions using trained models
4. **Message Queue System**: RabbitMQ for message distribution
5. **Task Processors**: Celery workers for different types of prediction processing
6. **Monitoring Tools**: Utilities to monitor message flow and system health

## Installation

### Prerequisites

- Python 3.11+
- RabbitMQ 4.0+
- Erlang

### Download Required Files

1. Download the model files:
   ```bash
   # Create models directory
   mkdir -p stock-prediction/models
   
   # Download models.zip from SharePoint
   curl -L "https://etsmtl365-my.sharepoint.com/:u:/g/personal/basile_paradis_1_ens_etsmtl_ca/EfknD8y0hDZFgWjIaiHWCdwBpK3YJvs8PIAC8RLRdMTfgw?e=UuDRZc" -o models.zip
   
   # Extract the contents to stock-prediction/models/
   unzip models.zip -d stock-prediction/models/
   
   # Clean up the zip file
   rm models.zip
   
   # This will create:
   # - stock-prediction/models/general/
   # - stock-prediction/models/prophet/
   # - stock-prediction/models/specific/
   ```

2. Download the data files:
   ```bash
   # Create data directory
   mkdir -p stock-prediction/data
   
   # Download data.zip from SharePoint
   curl -L "https://etsmtl365-my.sharepoint.com/:u:/g/personal/basile_paradis_1_ens_etsmtl_ca/EVr2YJqRv1lMlVSdDw_ZssIBQHkblF5_4tano1Fb9_9pBQ?e=Wvc8Gw" -o data.zip
   
   # Extract the contents to stock-prediction/data/
   unzip data.zip -d stock-prediction/data/
   
   # Clean up the zip file
   rm data.zip
   
   # This will create:
   # - stock-prediction/data/processed/
   # - stock-prediction/data/raw/
   ```

### Dependencies

```bash
pip install -r requirements.txt
```

### RabbitMQ Setup

Install and run RabbitMQ:

```bash
# macOS
brew install rabbitmq
CONF_ENV_FILE="/opt/homebrew/etc/rabbitmq/rabbitmq-env.conf" /opt/homebrew/opt/rabbitmq/sbin/rabbitmq-server -detached

# Check status (note the node name)
rabbitmqctl -n rabbitmq@localhost status
```

Enable the management plugin:

```bash
rabbitmq-plugins -n rabbitmq@localhost enable rabbitmq_management
```

Access the management interface at: http://localhost:15672 (user: guest, pass: guest)

## Data Collection and Processing

The system collects Google/Alphabet stock data using multiple sources with fallback mechanisms:

1. Primary: yfinance API
2. Secondary: pandas-datareader with Stooq
3. Fallback: Local historical data files

### Data Features

- Open, High, Low, Close prices
- Volume
- Additional derived features

### Preprocessing Steps

1. Handling missing data
2. Feature scaling using scikit-learn's StandardScaler
3. Sequence creation for time series modeling (sliding window approach)
4. Train/validate/test splits

## Model Training

### LSTM Model Architecture

Both TensorFlow and PyTorch implementations use similar LSTM architectures:

```
Input (sequence_length=60, features) → 
LSTM Layer (100 units) → 
Dropout (0.2) → 
Dense/Linear Output Layer (1)
```

### Training Process

1. Load and preprocess data
2. Create sequences with sliding window
3. Train LSTM models with Adam optimizer (learning_rate=0.001)
4. Monitor validation loss for early stopping
5. Save trained models and scalers

## Model Comparison

`torch_lstm_main.py` compares the performance of TensorFlow and PyTorch LSTM models:

### Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

### Visualizations

- Full time series predictions
- Close-up of most recent 50 days
- Predicted vs Actual scatter plot

## Prediction Service

`prediction_service.py` implements the stock price prediction service:

1. Fetches the latest available stock data
2. Applies preprocessing
3. Makes a prediction using the PyTorch LSTM model
4. Distributes the prediction to processing queues via Celery tasks

### Features

- Robust data fetching with multiple fallbacks
- Error handling for missing data
- Next business day calculation
- Resilient message publishing

## Message Queue System

The system uses RabbitMQ as the message broker with Celery for task processing:

### Components

- **Exchange**: stock_predictions (topic exchange)
- **Routing Key**: google.stock.prediction
- **Queues**:
  - prediction_queue_1: Basic processing
  - prediction_queue_2: Advanced analysis

### Message Format

```json
{
  "ticker": "GOOGL",
  "prediction_date": "2025-03-03",
  "prediction_time": "00:00:00",
  "predicted_open": 170.56,
  "model": "pytorch_lstm"
}
```

## Task Processing

`celery_tasks.py` defines the Celery tasks for processing predictions:

### Processor 1: Basic Processing

- Records the prediction
- Formats the output
- Could be extended to store in database, send notifications, etc.

### Processor 2: Advanced Analysis

- Calculates a price range (±5%)
- Determines confidence score
- More sophisticated analysis of prediction implications

### Celery Configuration

```python
app = Celery('stock_predictions',
            broker='amqp://guest:guest@localhost:5672//',
            backend='rpc://')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'celery_tasks.process_prediction': {'queue': 'prediction_queue_1'},
        'celery_tasks.analyze_prediction': {'queue': 'prediction_queue_2'}
    }
)
```

## Monitoring Tools

### RabbitMQ Monitor

`monitor_rabbitmq.py` provides a real-time visualization of the RabbitMQ system:

- Shows exchanges and their bindings
- Displays queue message counts
- Identifies queues with messages waiting for processing

### Task Status Checker

`check_task_status.py` allows checking the status of Celery tasks:

```bash
python check_task_status.py <task_id>
```

### System Runner

`run_prediction_system.py` provides an integrated way to run the complete system:

1. Starts a Celery worker
2. Runs the prediction service
3. Allows time for processing
4. Shows the results

## Usage

### Running the Complete System

```bash
python run_prediction_system.py
```

### Individual Components

**Model Comparison**:
```bash
python torch_lstm_main.py
```

**Prediction Service Only**:
```bash
python prediction_service.py
```

**Workers Only**:
```bash
celery -A celery_tasks worker -l info
```

**Monitor RabbitMQ**:
```bash
python monitor_rabbitmq.py
```

## Troubleshooting

### RabbitMQ Issues

If RabbitMQ fails to start or connect:

1. Check the node name: `rabbitmqctl status` may show a different node name
2. Use the correct node name: `rabbitmqctl -n rabbitmq@localhost status`
3. Run with full path: `CONF_ENV_FILE="/opt/homebrew/etc/rabbitmq/rabbitmq-env.conf" /opt/homebrew/opt/rabbitmq/sbin/rabbitmq-server`

### Celery Worker Issues

Celery workers may fail to start due to subprocess issues. Try:

```bash
celery -A celery_tasks worker -l info --pool=solo
```

### Data Fetching Issues

If data sources fail:

1. Check your internet connection
2. The system will automatically try alternative sources
3. Ensure you have the historical data files in the data directory

---

## Project Structure

```
stock-ai/
├── data/
│   ├── interim/              # Intermediate data
│   ├── processed/            # Processed and ready data
│   ├── raw/                  # Original data
│   └── script/               # Data collection scripts
├── model/                    # Model definitions
│   ├── ltsm_model.py         # PyTorch LSTM model
│   ├── dataset.py
│   ├── helpers.py
│   └── preprocessing.py
├── models/                   # Trained model files
│   ├── google_stock_price_lstm.model.keras
│   ├── google_stock_price_scaler.gz
│   ├── torch_lstm_model.pth
│   └── model_factory.py
├── notebooks/                # Jupyter notebooks
│   ├── 1-data-explanatory-analysis.ipynb
│   ├── 2-data-preprocessing.ipynb
│   └── 3-model-training.ipynb
├── reports/                  # Generated analysis
│   └── figures/              # Generated graphics
├── celery_tasks.py           # Celery task definitions
├── check_task_status.py      # Task status checker
├── monitor_rabbitmq.py       # RabbitMQ monitoring tool
├── prediction_consumer.py    # RabbitMQ consumer
├── prediction_service.py     # Prediction service
├── run_prediction_system.py  # Integrated system runner
├── torch_lstm_main.py        # Model comparison
└── train_torch_lstm.py       # PyTorch model training
```

## Features

* **Multi-horizon predictions** (next day, week)
* **Models with attention mechanisms** and long-term memory
* **Real-time data pipeline** with RabbitMQ
* Low-latency prediction serving
* **Robust message processing** with Celery
* Concept drift detection through monitoring tools

## Acknowledgments

- yfinance and Stooq for providing stock data
- TensorFlow and PyTorch for deep learning frameworks
- RabbitMQ and Celery for message queue and task processing