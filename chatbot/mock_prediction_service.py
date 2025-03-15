from flask import Flask, request, jsonify
import random
import datetime

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', 'UNKNOWN')
    
    # Generate random prediction data
    current_price = random.uniform(100, 500)
    prediction = current_price * (1 + random.uniform(-0.05, 0.05))
    
    response = {
        'ticker': ticker,
        'current_price': round(current_price, 2),
        'prediction': round(prediction, 2),
        'confidence_score': round(random.uniform(0.70, 0.95), 2),
        'model_type': 'RandomForest',
        'model_version': '1.0.0',
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "Mock Prediction Service Running"

if __name__ == '__main__':
    app.run(port=5001, debug=True) 