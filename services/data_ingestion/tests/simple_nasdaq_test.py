#!/usr/bin/env python3
"""
Simple test to show the raw response from get_nasdaq_stocks().
"""

import asyncio
import json
import sys
import os

# Add the project root to the Python path
# The script is in services/data_ingestion/tests, so we need to go up 3 levels
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)

from services.data_ingestion.data_service import DataService

async def simple_test():
    """Simple test to show the raw response."""
    
    print("🔍 Testing get_nasdaq_stocks() - Raw Response")
    print("=" * 50)
    
    try:
        # Initialize service
        data_service = DataService()
        
        # Get the data
        result = await data_service.get_nasdaq_stocks()
      
        
        # Show summary
        print("\n📈 SUMMARY:")
        print(f"Total stocks: {result.get('count', 'N/A')}")
        if 'data' in result and result['data']:
            print(f"First stock: {result['data'][0].get('symbol', 'N/A')}")
            print(f"Last stock: {result['data'][-1].get('symbol', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        await data_service.cleanup()

if __name__ == "__main__":
    asyncio.run(simple_test()) 
