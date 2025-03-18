import requests
import json

def test_sentiment_api():
    """Test the sentiment analysis API with different parameters"""
    print("Testing news analyzer API with different tickers...")
    
    tickers = ["AAPL", "MSFT", "GOOG"]
    
    for ticker in tickers:
        try:
            print(f"\nTesting ticker: {ticker}")
            # Use the key 'ticker' as shown in the screenshot
            response = requests.post('http://localhost:8092/api/analyze', 
                                    json={'ticker': ticker}, 
                                    timeout=5)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Received {len(data)} news items")
                if data:
                    print(f"First headline: {data[0].get('title', 'No title')}")
                    sentiment = data[0].get('sentiment', {})
                    print(f"Sentiment scores: {sentiment}")
            else:
                print(f"Error response: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
            
if __name__ == "__main__":
    test_sentiment_api() 