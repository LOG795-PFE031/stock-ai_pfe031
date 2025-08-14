# NotYahoo

A comprehensive financial analysis platform with AI-powered stock predictions, market sentiment analysis, and an interactive financial advisor chatbot.

## Features

- 📈 **Stock Prediction Service**: Get AI-generated predictions for stock performance
- 📰 **Market Sentiment Analysis**: Analyze market sentiment based on news articles
- 💬 **Interactive Chatbot**: Get answers to financial questions and personalized investment recommendations
- 📊 **Portfolio Management**: Track and manage your investments
- 📱 **Responsive UI**: Modern interface built with React and Chakra UI

## Tech Stack

- **Frontend**: React, Chakra UI, React Router
- **Charts & Visualization**: Recharts
- **Runtime**: Bun
- **AI & ML Libraries**: LangChain, OpenAI
- **HTTP Client**: Axios
- **Package Manager**: Bun

## Installation & Setup

### Prerequisites

- [Bun](https://bun.sh/) (latest version)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NotYahoo.git
   cd NotYahoo
   ```

2. Install dependencies:
   ```bash
   bun install
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   VITE_PREDICTION_SERVICE_URL=http://localhost:8000
   VITE_SENTIMENT_SERVICE_URL=http://localhost:8092
   VITE_OPENAI_API_KEY=your_openai_api_key
   ```

4. Start the development server:
   ```bash
   bun run dev
   ```

5. Open your browser and navigate to `http://localhost:5173`

## API Services

### Stock Prediction Service

Endpoint: `http://localhost:8000/docs/{ticker}`

Sample Response:
```json
{
  "symbol": "AAPL",
  "price": 150.25,
  "timestamp": "2024-03-21T00:00:00",
  "confidence_score": 0.85,
  "model_type": "lstm"
}
```

### Market Sentiment Analysis

Endpoint: `http://localhost:8092/analyze/{ticker}`

Sample Data Format:
```json
[
  {
    "url": "string",
    "title": "string",
    "ticker": "string",
    "content": "string",
    "date": "string",
    "sentiment_scores": {
      "positive": 0.1,
      "neutral": 0.2,
      "negative": 0.7
    }
  }
]
```

## API Client Implementation

The application uses a centralized API client service to interact with backend services:

- **Modular Architecture**: Each microservice has its own client with appropriate configuration
- **Type Safety**: All API requests and responses are fully typed
- **Error Handling**: Comprehensive error catching and reporting
- **Extensibility**: Generic methods make it easy to add new service endpoints

### Example Usage

```typescript
// Get stock prediction
const stockPrediction = await apiService.getStockPrediction('AAPL');

// Get sentiment analysis
const sentimentData = await apiService.getSentimentAnalysis('AAPL');

// Get historical data
const historicalData = await apiService.getHistoricalData('AAPL', '1y', '1d');
```

## Interactive Chatbot

The platform features an AI-powered chatbot that can:
- Answer questions about stock data and financial concepts
- Provide personalized investment recommendations
- Analyze user preferences and history to offer tailored advice

## Building for Production

```bash
bun run build
```

The built files will be in the `dist` directory.

## License

This project is licensed under the terms found in the LICENSE file.
