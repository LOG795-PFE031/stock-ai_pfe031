import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_sentiment_api():
    """Test the sentiment analysis API directly with multiple tickers"""
    print("Testing sentiment analysis API with real data...")
    
    # Get the URL from environment variable
    sentiment_url = os.getenv("SENTIMENT_SERVICE_URL", 'http://localhost:8092/api/analyze')
    print(f"Using service URL: {sentiment_url}")
    
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    
    for ticker in tickers:
        try:
            print(f"\n===== Testing ticker: {ticker} =====")
            start_time = time.time()
            
            # Make the request
            print(f"Sending request with payload: {{'ticker': '{ticker}'}}")
            response = requests.post(sentiment_url, 
                                     json={'ticker': ticker}, 
                                     timeout=30)  # Increased timeout for slow services
            
            elapsed = time.time() - start_time
            print(f"Request took {elapsed:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            # Check the response
            if response.status_code == 200:
                data = response.json()
                print(f"Received {len(data) if isinstance(data, list) else '(not a list)'} news items")
                
                if isinstance(data, list) and data:
                    # Print the first item details
                    print("\nFirst news item:")
                    for key, value in data[0].items():
                        if key == 'sentiment':
                            print(f"  sentiment scores: {value}")
                        else:
                            print(f"  {key}: {value}")
            else:
                print(f"Error response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Request timed out after 30 seconds")
        except Exception as e:
            print(f"Error: {e}")
            
if __name__ == "__main__":
    test_sentiment_api() 