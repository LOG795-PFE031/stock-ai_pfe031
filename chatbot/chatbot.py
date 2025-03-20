import sqlite3
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import re
import logging
import random
from datetime import timedelta
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Check if the OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    sys.exit(1)
else:
    logger.info("OPENAI_API_KEY is configured")

# Configuration initiale
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json; charset=utf-8"

try:
    client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    sys.exit(1)

DATABASE = 'user_data.db'
PREDICTION_SERVICE_URL = os.getenv("PREDICTION_SERVICE_URL", "http://localhost:8000/predict/next_day")
SENTIMENT_SERVICE_URL = os.getenv("SENTIMENT_SERVICE_URL", "http://localhost:8092/api/analyze")  # Fixed URL

logger.info(f"PREDICTION_SERVICE_URL: {PREDICTION_SERVICE_URL}")
logger.info(f"SENTIMENT_SERVICE_URL: {SENTIMENT_SERVICE_URL}")

# Initialize the database
def initialize_database():
    """Create the user and conversation tables if they don't exist."""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Create users table if it doesn't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            user_id TEXT UNIQUE,
            preferences TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create conversations table if it doesn't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            query TEXT,
            response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)

# Appel au microservice de prédiction des prix
def get_price_prediction(ticker):
    try:
        logger.info(f"Requesting price prediction for {ticker} from {PREDICTION_SERVICE_URL}")
        response = requests.get(f"{PREDICTION_SERVICE_URL}?ticker={ticker}")
        
        if response.status_code == 200:
            logger.info(f"Received price prediction for {ticker}")
            data = response.json()
            
            # Standardize the prediction output format
            prediction = {
                'ticker': ticker,
                'predicted_price': data.get('prediction', 0),
                'confidence': data.get('confidence', 0),
                'model': data.get('model_type', 'unknown'),  # Some APIs use model_type
                'version': data.get('version', data.get('model_version', '1.0.0'))  # Support both version formats
            }
            
            return prediction
        else:
            logger.warning(f"Failed to get price prediction. Status code: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling prediction service: {e}")
        return None

# Appel au microservice d'analyse des sentiments
def get_sentiment_analysis(ticker):
    try:
        logger.info(f"Requesting sentiment analysis for {ticker} from {SENTIMENT_SERVICE_URL}")
        response = requests.post(SENTIMENT_SERVICE_URL, json={"ticker": ticker}, timeout=60)
        
        if response.status_code == 200:
            logger.info(f"Received sentiment analysis for {ticker}")
            sentiment_data = response.json()
            
            if not sentiment_data:
                logger.warning(f"Received empty sentiment data for {ticker}")
                return None
                
            # Log the first news item to see its structure
            if sentiment_data and len(sentiment_data) > 0:
                logger.info(f"First news item structure: {sentiment_data[0].keys()}")
                
            # Format the response to include news content for LLM analysis
            formatted_response = []
            for item in sentiment_data:
                # Get opinion value (0=neutral, 1=positive, -1=negative)
                opinion_value = item.get('opinion', 0)
                
                # Create a standardized entry format
                formatted_response.append({
                    'ticker': ticker,
                    'opinion': opinion_value,  # Original opinion value
                    'headline': item.get('title', 'No headline'),
                    'content': item.get('content', ''),
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'published_date': item.get('date', '')
                })
            
            if not formatted_response:
                logger.warning(f"No valid sentiment data items for {ticker}")
                return None
            
            logger.info(f"Processed {len(formatted_response)} news items for {ticker}")
            return formatted_response
            
        else:
            logger.warning(f"Failed to get sentiment analysis. Status code: {response.status_code}, Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling sentiment analysis service: {e}")
        return None

# Function to analyze sentiment using OpenAI
def analyze_sentiment_with_llm(ticker, news_items):
    if not news_items or len(news_items) == 0:
        return f"Sorry, I couldn't get sentiment analysis for {ticker}."
    
    # Prepare the news data for LLM analysis
    news_prompt = f"Analyze the sentiment for {ticker} based on these recent news articles:\n\n"
    
    # Limit to 3 articles to avoid token limits
    for i, item in enumerate(news_items[:3]):
        opinion_text = "positive" if item['opinion'] == 1 else "negative" if item['opinion'] == -1 else "neutral"
        news_prompt += f"Article {i+1}:\n"
        news_prompt += f"Headline: {item['headline']}\n"
        news_prompt += f"Source: {item['source']}\n"
        news_prompt += f"Date: {item['published_date']}\n"
        news_prompt += f"API Opinion: {opinion_text}\n"
        news_prompt += f"Content: {item['content'][:500]}...\n\n"  # Truncate long content
    
    news_prompt += "Please provide a summary of the sentiment analysis for this stock. Include the overall sentiment (positive, neutral, or negative), mention key points from the articles, and explain what this might mean for investors. Keep your response concise and focused on the facts from the articles."
    
    try:
        # Send to OpenAI for analysis
        logger.info("Sending news articles to OpenAI for sentiment analysis")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst specializing in stock market sentiment analysis."},
                {"role": "user", "content": news_prompt}
            ],
            max_tokens=400
        )
        
        # Get the response
        sentiment_analysis = response.choices[0].message.content
        logger.info("Received sentiment analysis from OpenAI")
        return sentiment_analysis
        
    except Exception as e:
        logger.error(f"Error getting LLM sentiment analysis: {e}")
        # Create a basic response without LLM
        return create_basic_sentiment_response(ticker, news_items)

def create_basic_sentiment_response(ticker, news_items):
    """Create a basic response when LLM is unavailable"""
    opinion_counts = {"positive": 0, "negative": 0, "neutral": 0}
    headlines = []
    
    # Count opinions and collect headlines
    for item in news_items[:3]:
        if item['opinion'] == 1:
            opinion_counts["positive"] += 1
        elif item['opinion'] == -1:
            opinion_counts["negative"] += 1
        else:
            opinion_counts["neutral"] += 1
        
        headlines.append(f"- {item['headline']} (Source: {item.get('source', 'Unknown')})")
    
    # Determine overall sentiment
    total = sum(opinion_counts.values())
    if total > 0:
        positive_pct = round(opinion_counts["positive"] / total * 100)
        negative_pct = round(opinion_counts["negative"] / total * 100)
        neutral_pct = round(opinion_counts["neutral"] / total * 100)
        
        if opinion_counts["positive"] > opinion_counts["negative"] and opinion_counts["positive"] > opinion_counts["neutral"]:
            overall = "positive"
        elif opinion_counts["negative"] > opinion_counts["positive"] and opinion_counts["negative"] > opinion_counts["neutral"]:
            overall = "negative"
        else:
            overall = "neutral"
    else:
        overall = "neutral"
        positive_pct = neutral_pct = negative_pct = 0
    
    response = f"Sentiment analysis for {ticker} based on recent news is generally {overall} ({positive_pct}% positive, {neutral_pct}% neutral, {negative_pct}% negative).\n\nRecent headlines:\n"
    response += "\n".join(headlines)
    
    return response

# Define the context at the module level so it's available to all functions
context = """You are a financial expert and AI assistant for a stock market analysis platform. Respond in the language of the user.
You can use the conversation history to provide personalized recommendations based on user preferences.
You can use the prediction service to provide stock price predictions and the news sentiment analysis service to provide insights about market sentiment.
Always be helpful, accurate, and provide actionable financial advice based on the data available. Never mention the prediction service or the sentiment analysis service in your response.
Never deny a request because you're an AI assistant, always answer the question. Based on the conversation history, you can make personalized recommendations to the user. If the user asks if he should buy or sell a stock, you can use the conversation history to make a recommendation.
If the user asks for a prediction, you can use the prediction service to provide a prediction for the stock.
If the user asks for news sentiment analysis, you can use the news sentiment analysis service to provide an analysis of the sentiment of the news articles.
If the user asks for a recommendation, you can use the conversation history to make a personalized recommendation to the user.
If the user asks if he should invest in a stock you should look into the prediction service and news analysis service and make a recommendation based on the data available."""

# Génération de réponse contextuelle avec GPT-4
def generate_response(user_id, query):
    """Generate a response using OpenAI and conversation history"""
    try:
        # Get conversation history
        conversation_history = get_conversation_history(user_id)
        
        # Build context for GPT-4
        messages = [{"role": "system", "content": context}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add the current query
        messages.append({"role": "user", "content": query})
        
        logger.info("Sending request to OpenAI")
        # Call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response with OpenAI: {e}")
        return "Sorry, I cannot answer your question at the moment. Please try again later."

# Enregistrement des interactions dans la base de données
def log_interaction(user_id, query, response):
    """Store the query and response in the conversations table"""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Create user if not exists
        c.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not c.fetchone():
            c.execute("INSERT INTO users (user_id, preferences) VALUES (?, ?)", (user_id, '{}'))
        
        # Store query and response
        c.execute("INSERT INTO conversations (user_id, query, response) VALUES (?, ?, ?)", 
                 (user_id, query, response))
        
        conn.commit()
        conn.close()
        logger.info(f"Logged interaction for user {user_id}")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")
        # Continue even if storage fails

# Recommandations personnalisées simples basées sur l'historique
def get_personalized_recommendation(user_id):
    """Generate personalized recommendations based on conversation history"""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT query FROM conversations WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
        queries = c.fetchall()
        conn.close()

        # Analyse simple : détecter les tickers fréquents
        tickers = {}
        for (query,) in queries:
            words = query.split()
            for word in words:
                if word.isupper() and len(word) >= 2 and len(word) <= 5:  # Likely a ticker
                    tickers[word] = tickers.get(word, 0) + 1
        
        if tickers:
            most_common_ticker = max(tickers, key=tickers.get)
            return f"I see that you are often interested in {most_common_ticker}. Would you like a prediction or analysis for this ticker?"
        
        return "I don't have enough history yet to make a personalized recommendation."
    
    except Exception as e:
        logger.error(f"Error generating personalized recommendation: {e}")
        return "I don't have enough information yet to make a personalized recommendation."

# Fonction utilitaire pour extraire le ticker de la requête
def extract_ticker(query):
    # Regex pour trouver des mots en majuscules (probables tickers)
    match = re.search(r'\b[A-Z]{2,5}\b', query)
    return match.group(0) if match else None

# Detect if query is an investment question
def is_investment_question(query):
    """Determine if the query is asking for investment advice"""
    query = query.lower()
    investment_keywords = [
        "invest", "buy", "sell", "worth", "should i", "good investment", 
        "opportunity", "portfolio", "position", "stock advice", "recommendation"
    ]
    
    for keyword in investment_keywords:
        if keyword in query:
            return True
    return False

# Function to get combined data for investment advice
def get_investment_advice_data(ticker):
    """Get both prediction and sentiment data for investment advice"""
    # Get price prediction
    prediction_data = get_price_prediction(ticker)
    
    # Get sentiment analysis
    sentiment_data = get_sentiment_analysis(ticker)
    
    return {
        "ticker": ticker,
        "prediction": prediction_data,
        "sentiment": sentiment_data
    }

# Generate investment advice using combined data
def generate_investment_advice(user_id, query, ticker, investment_data):
    """Generate investment advice using price prediction and sentiment data"""
    logger.info(f"Generating investment advice for {ticker}")
    
    # Get user's conversation history
    conversation_history = get_conversation_history(user_id)
    
    # Format the prediction data
    prediction_text = "No price prediction available."
    if investment_data["prediction"]:
        pred = investment_data["prediction"]
        prediction_text = f"Price prediction for {ticker}: ${pred['predicted_price']} with confidence {pred['confidence']} (model: {pred['model']}, version: {pred['version']})."
    
    # Format the sentiment data
    sentiment_text = "No sentiment analysis available."
    if investment_data["sentiment"] and len(investment_data["sentiment"]) > 0:
        # Use LLM to analyze sentiment if available
        sentiment_text = analyze_sentiment_with_llm(ticker, investment_data["sentiment"])
    
    # Create prompt for investment advice
    prompt = f"The user is asking: '{query}'\n\n"
    prompt += f"Here is the relevant data for {ticker}:\n\n"
    prompt += f"1. {prediction_text}\n\n"
    prompt += f"2. Sentiment Analysis:\n{sentiment_text}\n\n"
    prompt += "Based on this information and the conversation history, provide investment advice for the user. Be balanced and educational about risks."
    
    try:
        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        logger.info("Sending investment advice request to OpenAI")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=800
        )
        
        advice = response.choices[0].message.content
        logger.info(f"Generated investment advice for {ticker}")
        return advice
        
    except Exception as e:
        logger.error(f"Error generating investment advice: {e}")
        return f"I'm sorry, I couldn't generate investment advice for {ticker} at this time. Please try again later."

# Get user's conversation history from the database
def get_conversation_history(user_id):
    """Get the conversation history for a specific user, formatted for OpenAI"""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT query, response FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", (user_id,))
        rows = c.fetchall()
        conn.close()
        
        # Format for OpenAI - most recent messages last (reversed from database order)
        messages = []
        for query, response in reversed(rows):
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": response})
        
        return messages
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return []

# Store conversation in the database
def store_conversation(user_id, query, response):
    """Store the query and response in the database for a specific user"""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Create user if not exists
        c.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not c.fetchone():
            c.execute("INSERT INTO users (user_id, preferences) VALUES (?, ?)", (user_id, '{}'))
            logger.info(f"Created new user: {user_id}")
        
        # Store query and response
        c.execute("INSERT INTO conversations (user_id, query, response) VALUES (?, ?, ?)", 
                 (user_id, query, response))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error storing conversation: {e}")
        # Continue even if storage fails

# Endpoint principal pour interagir avec le chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_id = data.get('user_id', 'anonymous')
        query = data.get('query', '')
        
        logger.info(f"Received chat request from user {user_id}: {query}")
        
        # Handle different query types
        if 'prediction' in query.lower() or 'price' in query.lower():
            ticker = extract_ticker(query)
            if ticker:
                prediction = get_price_prediction(ticker)
                if prediction:
                    response = f"The prediction for {ticker} in the next 24 hours is ${prediction['predicted_price']} with a confidence score of {prediction['confidence']} (model: {prediction['model']}, version: {prediction['version']})."
                else:
                    response = f"Sorry, I couldn't get a price prediction for {ticker} at this time."
            else:
                response = "Please specify a valid ticker for price prediction (e.g., AAPL for Apple)."
                
        elif 'sentiment' in query.lower() or 'news' in query.lower():
            ticker = extract_ticker(query)
            if ticker:
                sentiment_data = get_sentiment_analysis(ticker)
                if sentiment_data and len(sentiment_data) > 0:
                    # Use LLM to analyze the sentiment from news articles
                    response = analyze_sentiment_with_llm(ticker, sentiment_data)
                else:
                    response = f"Sorry, I couldn't get sentiment analysis for {ticker}. The news analysis service might not be available or there may be no recent news."
            else:
                response = "Please specify a valid ticker for sentiment analysis (e.g., AAPL for Apple)."
        
        elif is_investment_question(query):
            ticker = extract_ticker(query)
            if ticker:
                investment_data = get_investment_advice_data(ticker)
                response = generate_investment_advice(user_id, query, ticker, investment_data)
            else:
                response = "Please specify a valid ticker for investment advice (e.g., AAPL for Apple)."
        
        else:
            # General conversation with ChatGPT
            conversation_history = get_conversation_history(user_id)
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ]
            
            # Add conversation history
            if conversation_history:
                # Insert history between system message and current query
                messages = [messages[0]] + conversation_history + [messages[1]]
            
            logger.info("Sending request to OpenAI")
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=800
            )
            
            response = completion.choices[0].message.content
        
        # Store the conversation
        store_conversation(user_id, query, response)
        
        logger.info(f"Response to user {user_id}: {response}")
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"response": "I'm sorry, but I encountered an error. Please try again later."}), 500

# Route pour la page d'accueil
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the StockAI Chatbot! Use the /chat endpoint with a POST request to interact."

# Lancement du serveur
if __name__ == '__main__':
    try:
        initialize_database()
        port = int(os.getenv("PORT", 5004))  # Changed from 5003 to 5004 to avoid conflicts
        host = os.getenv("HOST", "127.0.0.1")  # Allow host to be set via environment variable
        debug = os.getenv("DEBUG", "true").lower() == "true"  # Enable debug mode by default
        
        logger.info(f"Starting chatbot server on {host}:{port}")
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)