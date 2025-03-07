import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import random

class NASDAQDataDownloader:
    def __init__(self, output_dir="data/raw", max_workers=5, base_delay=1):
        """
        Initialize the downloader with conservative settings
        
        Args:
            output_dir: Directory to save downloaded data
            max_workers: Maximum number of concurrent downloads
            base_delay: Base delay between requests in seconds
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.base_delay = base_delay
        self.logger = self._setup_logger()
        self.failed_downloads = set()

    def get_nasdaq_symbols(self) -> List[Tuple[str, str]]:
        """
        Get NASDAQ symbols and their sectors using Wikipedia data
        
        Returns:
            List of tuples containing (symbol, sector)
        """
        try:
            # Download NASDAQ-100 table from Wikipedia
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)
            
            # The NASDAQ-100 table is typically the first table
            df = tables[0]
            
            # Clean up the data
            df['Symbol'] = df['Symbol'].str.replace('^', '')  # Remove any ^ symbols
            df['Sector'] = df['Sector'].fillna('Unknown')  # Fill missing sectors
            
            # Extract symbols and sectors
            symbols = list(zip(df['Symbol'].tolist(), df['Sector'].tolist()))
            
            self.logger.info(f"Successfully retrieved {len(symbols)} NASDAQ-100 symbols")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching NASDAQ symbols: {str(e)}")
            return []

    def download_stock_data(self, symbol_info: Tuple[str, str], retries: int = 5) -> bool:
        """
        Download stock data with exponential backoff retry mechanism
        
        Args:
            symbol_info: Tuple of (symbol, sector)
            retries: Number of retry attempts
        """
        symbol, sector = symbol_info
        
        for attempt in range(retries):
            try:
                # Exponential backoff delay
                delay = self.base_delay * (2 ** attempt) + random.uniform(1, 3)
                time.sleep(delay)
                
                # Download historical stock data
                data = yf.download(
                    symbol,
                    start="2000-01-01",
                    end=datetime.now().strftime("%Y-%m-%d"),
                    auto_adjust=False,
                    progress=False
                )
                
                if data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    self.failed_downloads.add(symbol)
                    return False
                
                # Create sector-specific directory
                sector_dir = os.path.join(self.output_dir, sector.replace('/', '_'))
                os.makedirs(sector_dir, exist_ok=True)
                
                # Add metadata columns
                data = data.reset_index()
                data['Symbol'] = symbol
                data['Sector'] = sector
                
                # Save to sector-specific directory
                output_file = os.path.join(sector_dir, f"{symbol}_stock_price.csv")
                data.to_csv(output_file, index=False)
                
                # Also save a copy to a unified dataset directory for the general model
                unified_dir = os.path.join(self.output_dir, "unified")
                os.makedirs(unified_dir, exist_ok=True)
                unified_file = os.path.join(unified_dir, f"{symbol}_stock_price.csv")
                data.to_csv(unified_file, index=False)
                
                self.logger.info(f"Successfully downloaded data for {symbol}")
                return True
            
            except Exception as e:
                if "Too Many Requests" in str(e):
                    if attempt < retries - 1:
                        delay = self.base_delay * (2 ** attempt) + random.uniform(5, 10)
                        self.logger.warning(f"Rate limit hit for {symbol}, waiting {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        self.logger.error(f"Max retries reached for {symbol}")
                        self.failed_downloads.add(symbol)
                else:
                    self.logger.error(f"Error downloading {symbol}: {str(e)}")
                    if attempt == retries - 1:
                        self.failed_downloads.add(symbol)
            
            return False

    def download_batch(self, symbols: List[Tuple[str, str]], batch_size: int = 10) -> Tuple[int, int]:
        """Download a batch of symbols sequentially with delays"""
        successful = 0
        failed = 0
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            self.logger.info(f"Processing mini-batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
            
            for symbol_info in batch:
                if self.download_stock_data(symbol_info):
                    successful += 1
                else:
                    failed += 1
                
                self.logger.info(f"Progress: {successful + failed}/{len(symbols)} "
                               f"(Success: {successful}, Failed: {failed})")
        
        return successful, failed

    def download_all_stocks(self, batch_size: int = 10):
        """
        Download all NASDAQ-100 stocks in small batches
        
        Args:
            batch_size: Number of symbols to process in each batch
        """
        symbols = self.get_nasdaq_symbols()
        if not symbols:
            return
        
        successful, failed = self.download_batch(symbols, batch_size)
        
        # Report results
        self.logger.info(f"\nDownload completed:")
        self.logger.info(f"Successful downloads: {successful}")
        self.logger.info(f"Failed downloads: {failed}")
        
        # Save failed downloads to file
        if self.failed_downloads:
            self.logger.warning(f"Failed to download {len(self.failed_downloads)} symbols: {sorted(self.failed_downloads)}")
            failed_file = os.path.join(self.output_dir, "failed_downloads.txt")
            with open(failed_file, 'w') as f:
                f.write('\n'.join(sorted(self.failed_downloads)))
            self.logger.info(f"Failed downloads list saved to {failed_file}")
            
    def _setup_logger(self):
        logger = logging.getLogger('NASDAQDownloader')
        logger.setLevel(logging.INFO)
        
        # Add file handler
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        
        return logger

def main():
    downloader = NASDAQDataDownloader(
        output_dir="data/raw",
        max_workers=5,  # Reduced concurrent downloads
        base_delay=0    # Base delay between requests
    )
    downloader.download_all_stocks(batch_size=50)  # Smaller batch size

if __name__ == "__main__":
    main() 