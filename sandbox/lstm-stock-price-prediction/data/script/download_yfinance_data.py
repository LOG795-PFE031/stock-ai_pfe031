import yfinance as yf

# Define the stock ticker and time period
ticker_symbol = "GOOG"  # Google's stock ticker
start_date = "2000-01-01"  # Start date of historical data
end_date = "2025-02-09"  # End date (today's date)

# Download historical stock data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Save the data to a CSV file
output_file = "2025_google_stock_price_full.csv"
data.to_csv(output_file)

print(f"Historical data for {ticker_symbol} has been saved to {output_file}.")
