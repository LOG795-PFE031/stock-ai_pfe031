import requests
import time
import json

def get_sentiment_analysis_direct(ticker):
    """Test direct connection to sentiment analysis API and format the results"""
    sentiment_url = "http://localhost:8092/api/analyze"
    print(f"Requesting sentiment analysis for {ticker} from {sentiment_url}")
    
    try:
        start_time = time.time()
        response = requests.post(sentiment_url, json={"ticker": ticker}, timeout=60)
        elapsed = time.time() - start_time
        print(f"Request took {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            print(f"Status code: {response.status_code}")
            sentiment_data = response.json()
            
            if not sentiment_data or len(sentiment_data) == 0:
                print(f"Received empty sentiment data for {ticker}")
                return None
            
            print(f"Received {len(sentiment_data)} news items")
            
            # Print the raw data structure of the first item to understand format
            if sentiment_data and len(sentiment_data) > 0:
                print("\nRaw data format of first item:")
                first_item = sentiment_data[0]
                for key, value in first_item.items():
                    print(f"  {key}: {value}")
                
                print("\nChecking for sentiment structure:")
                if 'sentiment' in first_item:
                    print("  'sentiment' key exists")
                else:
                    print("  'sentiment' key does NOT exist")
                    
                    # Check if 'opinion' exists instead
                    if 'opinion' in first_item:
                        print("  'opinion' key exists with value:", first_item['opinion'])
            
            # Format the response to match the expected structure for the chatbot
            formatted_response = []
            for item in sentiment_data:
                # Try both 'sentiment' and 'opinion' fields
                sentiment_scores = {}
                if 'sentiment' in item:
                    sentiment_scores = item.get('sentiment', {})
                elif 'opinion' in item:
                    # Map opinion (0, 1, -1) to sentiment scores
                    opinion = item.get('opinion', 0)
                    if opinion == 1:  # Positive
                        sentiment_scores = {'positive': 0.8, 'neutral': 0.1, 'negative': 0.1}
                    elif opinion == -1:  # Negative
                        sentiment_scores = {'positive': 0.1, 'neutral': 0.1, 'negative': 0.8}
                    else:  # Neutral
                        sentiment_scores = {'positive': 0.2, 'neutral': 0.6, 'negative': 0.2}
                else:
                    # Skip items without sentiment info
                    continue
                
                # Create a standardized entry format
                formatted_response.append({
                    'ticker': ticker,
                    'sentiment_scores': {
                        'positive': sentiment_scores.get('positive', 0.0),
                        'negative': sentiment_scores.get('negative', 0.0),
                        'neutral': sentiment_scores.get('neutral', 0.0)
                    },
                    'headline': item.get('title', 'No headline'),
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'published_date': item.get('date', '')
                })
            
            if not formatted_response:
                print(f"No valid sentiment data items for {ticker}")
                return None
            
            print(f"Processed {len(formatted_response)} news items for {ticker}")
            
            # Format the response for display
            format_sentiment_response(ticker, formatted_response)
            
            return formatted_response
            
        else:
            print(f"Failed to get sentiment analysis. Status code: {response.status_code}, Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling sentiment analysis service: {e}")
        return None

def format_sentiment_response(ticker, sentiment_data):
    """Format the sentiment data into a readable response"""
    if not sentiment_data or len(sentiment_data) == 0:
        print(f"Sorry, I couldn't get sentiment analysis for {ticker}.")
        return
        
    summary = {"positive": 0, "negative": 0, "neutral": 0}
    headlines = []
    
    # Calculate average sentiment and collect headlines
    for item in sentiment_data[:3]:  # Limit to first 3 news items for the response
        scores = item['sentiment_scores']
        summary["positive"] += scores['positive']
        summary["negative"] += scores['negative']
        summary["neutral"] += scores['neutral']
        headlines.append(f"- {item['headline']} (Source: {item.get('source', 'Unknown')})")
    
    count = min(len(sentiment_data), 3)
    if count > 0:
        summary["positive"] = round(summary["positive"] / count, 2)
        summary["negative"] = round(summary["negative"] / count, 2)
        summary["neutral"] = round(summary["neutral"] / count, 2)
    
    response = f"Sentiment analysis for {ticker} based on recent news shows: positive {summary['positive']}, neutral {summary['neutral']}, negative {summary['negative']}.\n\nRecent headlines:\n"
    response += "\n".join(headlines)
    
    print("\nFormatted response:")
    print(response)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
    
    for ticker in tickers:
        print(f"\n===== Testing ticker: {ticker} =====")
        sentiment_data = get_sentiment_analysis_direct(ticker)
        print("-" * 50) 