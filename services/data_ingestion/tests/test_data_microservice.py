#!/usr/bin/env python3
"""
Test script for the data microservice.
"""

import requests
import json
import time

def test_data_service():
    """Test the data service endpoints."""
    
    base_url = "http://localhost:8001"
    
    print("Testing Data Service Microservice...")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    # Test health check
    print("\n1. Testing health check...")
    # The container need to be running for this test to pass
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test welcome endpoint
    print("\n2. Testing welcome endpoint...")
    try:
        response = requests.get(f"{base_url}/data/welcome")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test stocks list endpoint
    print("\n3. Testing stocks list endpoint...")
    try:
        response = requests.get(f"{base_url}/data/stocks")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test current price endpoint
    print("\n4. Testing current price endpoint...")
    try:
        response = requests.get(f"{base_url}/data/stock/current/?symbol=AAPL")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test recent data endpoint
    print("\n5. Testing recent data endpoint...")
    try:
        response = requests.get(f"{base_url}/data/stock/recent/?symbol=AAPL&days_back=5")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Symbol: {data.get('symbol')}")
        print(f"Total records: {data.get('total_records')}")
        print(f"Stock info: {data.get('stock_info')}")
        if data.get('prices'):
            print(f"First price record: {data['prices'][0]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test historical data endpoint
    print("\n6. Testing historical data endpoint...")
    try:
        response = requests.get(f"{base_url}/data/stock/historical/?symbol=AAPL&start_date=2024-01-01&end_date=2024-01-31")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Symbol: {data.get('symbol')}")
        print(f"Total records: {data.get('total_records')}")
        if data.get('prices'):
            print(f"First price record: {data['prices'][0]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test from-end-date endpoint
    print("\n7. Testing from-end-date endpoint...")
    try:
        response = requests.get(f"{base_url}/data/stock/from-end-date/?symbol=AAPL&end_date=2024-01-31&days_back=5")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Symbol: {data.get('symbol')}")
        print(f"Total records: {data.get('total_records')}")
        if data.get('prices'):
            print(f"First price record: {data['prices'][0]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test metrics endpoint
    print("\n8. Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics")
        print(f"Status: {response.status_code}")
        print("Metrics available (first 500 chars):")
        print(response.text[:500])
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_data_service() 
