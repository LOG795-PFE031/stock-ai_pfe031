from flask import Flask, request, jsonify
import random
import datetime

app = Flask(__name__)

@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.json
    ticker = data.get('ticker', 'UNKNOWN')
    
    # Generate random sentiment scores
    positive = round(random.uniform(0.2, 0.7), 2)
    negative = round(random.uniform(0.0, 0.3), 2)
    neutral = round(1 - positive - negative, 2)
    
    # Ensure scores sum to 1.0
    total = positive + negative + neutral
    positive /= total
    negative /= total
    neutral /= total
    
    response = [{
        'ticker': ticker,
        'sentiment_scores': {
            'positive': round(positive, 2),
            'negative': round(negative, 2),
            'neutral': round(neutral, 2)
        },
        'articles_analyzed': random.randint(3, 10),
        'timestamp': datetime.datetime.now().isoformat()
    }]
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "Mock Sentiment Analysis Service Running"

if __name__ == '__main__':
    app.run(port=5002, debug=True) 