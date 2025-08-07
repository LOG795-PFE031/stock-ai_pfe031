import asyncio
import httpx
from time import perf_counter

# Configuration
BASE_URL = "http://localhost:8000"  # Change this if needed
ENDPOINT = "/api/data/stock/current"
SYMBOLS = [
    # Tech stocks
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NFLX", "NVDA", "TSLA", 
    "INTC", "AMD", "IBM", "ORCL", "CRM", "ADBE", "CSCO", 
    
    # Financial stocks
    "JPM", "BAC", "WFC", "C", "GS", "V", "MA", "AXP", 
    
    # Consumer goods & retail
    "PG", "KO", "PEP", "WMT", "TGT", "COST", "MCD", 
    
    # Healthcare
    "JNJ", "PFE", "MRK", "UNH", "ABBV", 
    
    # Other sectors
    "DIS", "VZ", "T", "XOM", "CVX"
]  # 40 Yahoo Finance compatible stock symbols

# Number of concurrent requests to fire (one for each symbol)
NUM_REQUESTS = len(SYMBOLS)


# Fetch stock data for a given symbol
async def fetch_stock_data(client, symbol):
    try:
        params = {"symbol": symbol}
        response = await client.get(f"{BASE_URL}{ENDPOINT}", params=params)
        return response
    except Exception as e:
        print(f"Request failed for {symbol}: {e}")
        return None


# Main function to handle concurrent requests
async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        # Create a task for each symbol
        tasks = [fetch_stock_data(client, symbol) for symbol in SYMBOLS]

        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks)

        # Optional: Count successes/failures
        success = sum(1 for r in responses if r is not None and r.status_code == 200)
        print(f"\n✅ {success}/{NUM_REQUESTS} requests succeeded.")


if __name__ == "__main__":
    print(f"# Request : {NUM_REQUESTS}")
    t = perf_counter()
    asyncio.run(main())
    print(f"Time taken: {perf_counter() - t:.2f} seconds")
