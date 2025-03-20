
# Stock-AI: Advanced Stock Prediction and Analysis System

A comprehensive platform for stock price prediction and sentiment analysis using deep learning models (TensorFlow and PyTorch LSTM), distributed processing with RabbitMQ, and an AI-powered chatbot interface.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Running the System](#running-the-system)
5. [Testing the Services](#testing-the-services)
6. [Troubleshooting](#troubleshooting)
7. [Features](#features)
8. [Project Structure](#project-structure)

## Overview

The Stock-AI system is an integrated platform that:

- Performs stock price predictions using deep learning models
- Analyzes news sentiment to provide investment insights
- Processes data through distributed message queues
- Provides an interactive chatbot interface for user queries
- Offers microservice architecture for scalability and resilience

The system combines multiple technologies including PyTorch, TensorFlow, RabbitMQ, and Docker to create a comprehensive stock analysis platform.

## System Architecture

The system consists of the following components:

1. **Stock Prediction Service**: Predicts future stock prices using LSTM models
2. **News Analyzer Service**: Performs sentiment analysis on financial news
3. **RabbitMQ Message Queue**: Handles distributed message processing
4. **Chatbot Interface**: Provides natural language interaction with the system
5. **StockAI Backend**: C# service that manages authentication and API coordination

All components are containerized using Docker for easy deployment and scaling.

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Git

### Step 1: Download Required Models

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

### Step 2: Set Up the StockAI Backend

1. Clone the stockai-backend repository (C# service with RabbitMQ)
2. Follow the README steps in that repository to get it up and running
3. Ensure the RabbitMQ container is running properly

### Step 3: Create and Configure Docker Network

Create a Docker network to connect all services:

```bash
docker network create auth
```

Connect all required services to the network:

```bash
docker network connect auth rabbitmq && docker network connect auth stock-predictor && docker network connect auth news-analyzer && docker network connect auth monolith
```

### Step 4: Configure the Chatbot

Create a `.env` file in the `chatbot` folder with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Running the System

### Build and Start the Services

Build and run all services with Docker Compose:

```bash
docker compose up --build
```

**Note**: Initial build may take 10-30 minutes as PyTorch and other dependencies are installed.

### Starting Individual Components

If you need to start components separately:

**Stock Prediction Service**:
```bash
docker compose up stock-predictor
```

**News Analyzer Service**:
```bash
docker compose up news-analyzer
```

**Chatbot (locally)**:
```bash
cd chatbot
python chatbot.py
```

## Testing the Services

### Stock Prediction Service

Access the Swagger UI documentation and test interface:
```
http://localhost:8000/docs
```

Example API calls:
```bash
curl -X GET "http://localhost:8000/predict/AAPL"
```

### News Sentiment Analysis Service

Access the sentiment analysis service:
```
http://localhost:8092/
```

Example API calls:
```bash
curl -X GET "http://localhost:8092/api/sentiment/AAPL"
```

### Chatbot Interface

Test the chatbot with a curl command:

```bash
curl -X POST http://localhost:5004/chat -H "Content-Type: application/json" -d '{"user_id": "test_user", "query": "Should I invest into MSFT?"}'
```

Example queries for the chatbot:
- "What's the price prediction for AAPL?"
- "Show me the sentiment analysis for Tesla"
- "Should I invest in Google right now?"
- "What's the news sentiment for NVDA?"

## Troubleshooting

### Docker Networking Issues

If services cannot communicate:
1. Check if all containers are on the same network:
   ```bash
   docker network inspect auth
   ```
2. Restart the network connection if needed:
   ```bash
   docker network disconnect auth container_name && docker network connect auth container_name
   ```

### RabbitMQ Connection Problems

If services can't connect to RabbitMQ:
1. Check RabbitMQ status:
   ```bash
   docker exec -it rabbitmq rabbitmqctl status
   ```
2. Ensure the RabbitMQ management interface is accessible:
   ```
   http://localhost:15672 (user: guest, pass: guest)
   ```

### Model Loading Errors

If the prediction service fails to load models:
1. Verify the models were correctly placed in the `stock-prediction` folder
2. Check container logs:
   ```bash
   docker logs stock-predictor
   ```

### Chatbot API Key Issues

If the chatbot fails to connect to OpenAI:
1. Verify your API key in the `.env` file
2. Check for OpenAI rate limits or API changes

## Features

* **Multi-model Stock Prediction**: TensorFlow and PyTorch LSTM models
* **News Sentiment Analysis**: NLP-based analysis of financial news
* **AI-Powered Chatbot**: Natural language interface using OpenAI
* **Real-time Data Pipeline**: RabbitMQ for distributed processing
* **Docker Containerization**: Easy deployment and scaling
* **Microservice Architecture**: Independent, loosely-coupled services

## Project Structure

```
stock-ai/
├── stock-prediction/         # Stock prediction service
│   ├── models/               # Trained ML models
│   ├── data/                 # Stock data
│   └── logs/                 # Service logs
├── news-analyzer/            # News sentiment analysis service
│   ├── Dockerfile
│   └── ...
├── chatbot/                  # AI chatbot interface
│   ├── chatbot.py            # Main chatbot code
│   └── .env                  # Environment variables (API keys)
├── docker-compose.yml        # Docker compose configuration
└── README.md                 # This file
```

## Acknowledgments

- OpenAI for the chatbot capabilities
- yfinance and Stooq for providing stock data
- TensorFlow and PyTorch for deep learning frameworks
- RabbitMQ for message queue processing
- The open-source community for various libraries and tools

---

## License

[MIT License](LICENSE)
