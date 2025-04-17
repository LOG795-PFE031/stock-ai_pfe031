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
5. **Backend**: C# service that manages authentication and API coordination

All components are containerized using Docker for easy deployment and scaling.

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Git

### Step 1: Set Up Data and Models

1. Create the required directories:

   ```bash
   mkdir -p data/{stock,news,processed,raw}
   mkdir -p models/{general,prophet,specific}
   mkdir -p logs
   ```
2. Download the required data and models:

   - Option 1: Manual Download

     1. Visit the SharePoint links in your browser:
        - Data: https://etsmtl365-my.sharepoint.com/:u:/g/personal/basile_paradis_1_ens_etsmtl_ca/Ed_0wI8AD8xNmEtrGtQpTRAB_WuEg9C99yxiq7mraEsr3Q?e=uPbi7a
        - Models: https://etsmtl365-my.sharepoint.com/:u:/g/personal/basile_paradis_1_ens_etsmtl_ca/EUR5xD1QhJNAoJsduDJhJLYBxRIYmJGSe3J5fBIuTPpaLw?e=PlasAh
     2. Download the files
     3. Extract the contents to the root of the project
   - Option 2: Using the Setup Script

     ```bash
     cd scripts
     pip install -r requirements.txt
     python setup_data.py
     ```
3. Verify the data structure:

   ```bash
   # Check data directories
   ls -la data/stock
   ls -la data/news

   # Check model directories
   ls -la models/general
   ls -la models/prophet
   ls -la models/specific
   ```

### Step 2: Set Up the Backend

1. Clone the backend repository (C# service with RabbitMQ)
2. Follow the README steps in that repository to get it up and running
3. Ensure the RabbitMQ container is running properly

### Step 3: Create and Configure Docker Network

The system uses the "stockai-backend_auth" network created by the Backend. No manual network creation is needed, but you need to ensure your services are configured to use this network.

1. First, verify the network exists after starting the backend services:

```bash
docker network ls | grep stockai-backend_auth
```

2. Update your docker-compose.yml to use the existing network:

```yaml
services:
  # Your services configuration...
  networks:
    - stockai-backend_auth

networks:
  stockai-backend_auth:
    external: true
```

3. To verify network connectivity, you can test the connection (after starting the services):

```bash
# Install netcat in the container
docker exec -it stock-predictor apt-get update && docker exec -it stock-predictor apt-get install -y netcat-openbsd

# Test connection to RabbitMQ
docker exec stock-predictor nc -zv rabbitmq 5672
```

A successful connection will show: "Connection to rabbitmq port [tcp/amqp] succeeded!"

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
curl -X GET "http://localhost:8000/api/predict/AAPL"
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

### Data and Model Issues

If you encounter issues with data or models:

1. Verify the data structure:

   ```bash
   # Check if directories exist and have content
   ls -la data/stock
   ls -la data/news
   ls -la models/general
   ```
2. If directories are empty:

   - Try downloading the data manually from SharePoint
   - Make sure you're logged into your ETS account
   - Check if the SharePoint links are still valid
3. If you get HTML instead of zip files:

   - Make sure you're logged into your ETS account
   - Try opening the links in a new browser window
   - Use the "Download" button in SharePoint instead of direct links

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
