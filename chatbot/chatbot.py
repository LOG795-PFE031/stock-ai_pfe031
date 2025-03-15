import sqlite3
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Check if the OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    exit(1)
else:
    logger.info("OPENAI_API_KEY is configured")

# Configuration initiale
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json; charset=utf-8"

try:
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    exit(1)

DATABASE = 'user_data.db'
PREDICTION_SERVICE_URL = os.getenv("PREDICTION_SERVICE_URL", 'http://localhost:5001/predict')
SENTIMENT_SERVICE_URL = os.getenv("SENTIMENT_SERVICE_URL", 'http://localhost:5002/sentiment')

logger.info(f"PREDICTION_SERVICE_URL: {PREDICTION_SERVICE_URL}")
logger.info(f"SENTIMENT_SERVICE_URL: {SENTIMENT_SERVICE_URL}")

# Initialisation de la base de données SQLite
def init_db():
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            # Table for users and their preferences
            c.execute('''CREATE TABLE IF NOT EXISTS users
                         (user_id TEXT PRIMARY KEY, preferences TEXT)''')
            # Table for interaction history
            c.execute('''CREATE TABLE IF NOT EXISTS interactions
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, timestamp TEXT, query TEXT, response TEXT)''')
            conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

# Appel au microservice de prédiction des prix
def get_price_prediction(ticker):
    try:
        logger.info(f"Requesting price prediction for {ticker} from {PREDICTION_SERVICE_URL}")
        response = requests.post(PREDICTION_SERVICE_URL, json={'ticker': ticker}, timeout=5)
        if response.status_code == 200:
            logger.info(f"Received price prediction for {ticker}")
            return response.json()
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
        response = requests.post(SENTIMENT_SERVICE_URL, json={'ticker': ticker}, timeout=5)
        if response.status_code == 200:
            logger.info(f"Received sentiment analysis for {ticker}")
            return response.json()
        else:
            logger.warning(f"Failed to get sentiment analysis. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling sentiment service: {e}")
        return None

# Génération de réponse contextuelle avec GPT-4
def generate_response(user_id, query):
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            # Récupérer les 5 dernières interactions pour le contexte
            c.execute("SELECT query, response FROM interactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 5", (user_id,))
            history = c.fetchall()

        # Construire le contexte pour GPT-4
        context = "Vous êtes un assistant financier expert. Voici l'historique récent de la conversation :\n"
        for q, r in reversed(history):
            context += f"Utilisateur : {q}\nAssistant : {r}\n"
        context += f"Utilisateur : {query}\nAssistant :"

        logger.info("Sending request to OpenAI")
        # Appel à GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": context}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response with OpenAI: {e}")
        return "Désolé, je ne peux pas répondre à votre question en ce moment. Veuillez réessayer plus tard."

# Enregistrement des interactions dans la base de données
def log_interaction(user_id, query, response):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO interactions (user_id, timestamp, query, response) VALUES (?, ?, ?, ?)",
                  (user_id, timestamp, query, response))
        conn.commit()

# Recommandations personnalisées simples basées sur l'historique
def get_personalized_recommendation(user_id):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT query FROM interactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
        queries = c.fetchall()

    # Analyse simple : détecter les tickers fréquents
    tickers = {}
    for (query,) in queries:
        words = query.split()
        for word in words:
            if word.isupper() and len(word) > 1:  # Supposition qu'un ticker est en majuscules
                tickers[word] = tickers.get(word, 0) + 1
    if tickers:
        most_common_ticker = max(tickers, key=tickers.get)
        return f"Je vois que vous vous intéressez souvent à {most_common_ticker}. Voulez-vous une prédiction ou une analyse pour ce ticker ?"
    return "Je n'ai pas encore assez d'historique pour vous faire une recommandation personnalisée."

# Fonction utilitaire pour extraire le ticker de la requête
def extract_ticker(query):
    # Regex pour trouver des mots en majuscules (probables tickers)
    match = re.search(r'\b[A-Z]{2,5}\b', query)
    return match.group(0) if match else None

# Endpoint principal pour interagir avec le chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_id = data.get('user_id')
        query = data.get('query')

        logger.info(f"Received chat request from user {user_id}: {query}")

        if not user_id or not query:
            return jsonify({'error': 'user_id et query sont requis'}), 400

        # Vérifier ou créer l'utilisateur
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("SELECT preferences FROM users WHERE user_id=?", (user_id,))
            user = c.fetchone()
            if not user:
                c.execute("INSERT INTO users (user_id, preferences) VALUES (?, ?)", (user_id, ''))
                conn.commit()
                logger.info(f"Created new user: {user_id}")

        # Traitement de la requête
        response = ""
        if 'prédiction' in query.lower() or 'prix' in query.lower():
            ticker = extract_ticker(query)
            if ticker:
                prediction = get_price_prediction(ticker)
                if prediction:
                    response = (f"La prédiction pour {ticker} dans les prochaines 24 heures est de {prediction['prediction']} USD "
                              f"avec un score de confiance de {prediction['confidence_score']} (modèle : {prediction['model_type']}, "
                              f"version : {prediction['model_version']}).")
                else:
                    response = f"Désolé, je n'ai pas pu obtenir la prédiction pour {ticker}. Le service de prédiction n'est peut-être pas disponible."
            else:
                response = "Veuillez spécifier un ticker valide pour la prédiction (par exemple, AAPL pour Apple)."
        elif 'sentiment' in query.lower() or 'actualité' in query.lower():
            ticker = extract_ticker(query)
            if ticker:
                sentiment = get_sentiment_analysis(ticker)
                if sentiment and len(sentiment) > 0:
                    scores = sentiment[0]['sentiment_scores']
                    response = (f"L'analyse des sentiments pour {ticker} basée sur les actualités montre : "
                              f"positif {scores['positive']}, neutre {scores['neutral']}, négatif {scores['negative']}.")
                else:
                    response = f"Désolé, je n'ai pas pu obtenir l'analyse des sentiments pour {ticker}. Le service d'analyse n'est peut-être pas disponible."
            else:
                response = "Veuillez spécifier un ticker valide pour l'analyse des sentiments (par exemple, AAPL pour Apple)."
        elif 'recommandation' in query.lower():
            response = get_personalized_recommendation(user_id)
        else:
            # Réponse générale avec GPT-4
            response = generate_response(user_id, query)

        # Enregistrer l'interaction
        log_interaction(user_id, query, response)
        logger.info(f"Response to user {user_id}: {response}")

        # Avant de retourner la réponse, assurez-vous que les en-têtes sont correctement définis
        response_data = {'response': response}
        resp = jsonify(response_data)
        resp.headers['Content-Type'] = 'application/json; charset=utf-8'
        return resp
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e), 'response': "Une erreur est survenue. Veuillez réessayer."}), 500

# Route pour la page d'accueil
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Chatbot! Use the /chat endpoint with a POST request to interact."

# Lancement du serveur
if __name__ == '__main__':
    try:
        init_db()
        port = 5003  # Changer le port pour éviter les conflits avec AirPlay sur macOS
        logger.info(f"Starting chatbot server on port {port}")
        app.run(port=port, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")