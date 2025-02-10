import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import os

def get_sp500_symbols():
    """Get S&P 500 symbols using Wikipedia data"""
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df['Symbol'].tolist()

def download_stock_data(ticker_symbol, start_date, end_date, output_dir):
    """Download stock data for a single symbol with error handling"""
    try:
        # Download historical stock data
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if data.empty:
            print(f"No data available for {ticker_symbol}")
            return False
            
        # Create filename
        output_file = os.path.join(output_dir, f"{ticker_symbol}_stock_price.csv")
        
        # Save the data to a CSV file
        data.to_csv(output_file)
        print(f"Successfully downloaded data for {ticker_symbol}")
        return True
        
    except Exception as e:
        print(f"Error downloading {ticker_symbol}: {str(e)}")
        return False

def main():
    # Create output directory if it doesn't exist
    output_dir = "stock_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define time period
    start_date = "2000-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Get S&P 500 symbols
    symbols = get_sp500_symbols()
    print(f"Found {len(symbols)} symbols to process")
    
    # Download data for each symbol
    successful = 0
    failed = 0
    
    for symbol in symbols:
        # Add delay to prevent hitting rate limits
        time.sleep(1)
        
        if download_stock_data(symbol, start_date, end_date, output_dir):
            successful += 1
        else:
            failed += 1
            
        print(f"Progress: {successful + failed}/{len(symbols)} (Success: {successful}, Failed: {failed})")

if __name__ == "__main__":
    main()