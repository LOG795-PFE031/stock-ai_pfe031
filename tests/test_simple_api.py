"""
Simple test to check API endpoints results.
"""

import httpx
import asyncio
from datetime import datetime, timedelta

# Base URL for your API
BASE_URL = "http://localhost:8000/api"  # Change this to your API URL

async def test_endpoints():
    """Test all data service endpoints."""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🧪 Testing API Data Service Endpoints")
        print(f"🌐 Base URL: {BASE_URL}")
        print("=" * 60)
        
        # Test 0: Basic connectivity test
        print("0️⃣ Testing basic connectivity")
        try:
            response = await client.get(f"{BASE_URL}/welcome")
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ API server is reachable!")
            else:
                print("   ⚠️  API server responding but not with expected status")
        except Exception as e:
            print(f"   ❌ Cannot reach API server: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")
            print("   💡 Make sure your API server is running!")
        
        print()
        
        # Test 1: Get stocks list
        print("1️⃣ Testing GET /data/stocks")
        print(f"   🔗 URL: {BASE_URL}/data/stocks")
        try:
            response = await client.get(f"{BASE_URL}/data/stocks")
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Found {data.get('count', 0)} stocks")
                if data.get('data'):
                    print(f"   📊 First stock: {data['data'][0].get('symbol')} - {data['data'][0].get('companyName')}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
                print(f"   📄 Response Headers: {dict(response.headers)}")
                print(f"   📝 Response Text: {response.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")
        
        print()
        
        # Test 2: Get current stock data
        print("2️⃣ Testing GET /data/stock/current")
        print(f"   🔗 URL: {BASE_URL}/data/stock/current?symbol=AAPL")
        try:
            response = await client.get(f"{BASE_URL}/data/stock/current?symbol=AAPL")
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Current price for {data.get('symbol')}: ${data.get('data', ['N/A'])[0]}")
                print(f"   📈 Stock name: {data.get('name')}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
                print(f"   📄 Response Headers: {dict(response.headers)}")
                print(f"   📝 Response Text: {response.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")
        
        print()
        
        # Test 3: Get historical stock data
        print("3️⃣ Testing GET /data/stock/historical")
        start_date = (datetime.now() - timedelta(days=7)).isoformat()
        end_date = datetime.now().isoformat()
        historical_url = f"{BASE_URL}/data/stock/historical?symbol=AAPL&start_date={start_date}&end_date={end_date}"
        print(f"   🔗 URL: {historical_url}")
        try:
            response = await client.get(historical_url)
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Historical data for {data.get('symbol')}")
                print(f"   📊 Data points: {len(data.get('data', []))}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
                print(f"   📄 Response Headers: {dict(response.headers)}")
                print(f"   📝 Response Text: {response.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")
        
        print()
        
        # Test 4: Get recent stock data
        print("4️⃣ Testing GET /data/stock/recent")
        recent_url = f"{BASE_URL}/data/stock/recent?symbol=AAPL&days_back=5"
        print(f"   🔗 URL: {recent_url}")
        try:
            response = await client.get(recent_url)
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Recent data for {data.get('symbol')}")
                print(f"   📊 Data points: {len(data.get('data', []))}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
                print(f"   📄 Response Headers: {dict(response.headers)}")
                print(f"   📝 Response Text: {response.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")
        
        print()
        
        # Test 5: Get stock data from end date
        print("5️⃣ Testing GET /data/stock/from-end-date")
        end_date = datetime.now().isoformat()
        from_end_url = f"{BASE_URL}/data/stock/from-end-date?symbol=AAPL&end_date={end_date}&days_back=3"
        print(f"   🔗 URL: {from_end_url}")
        try:
            response = await client.get(from_end_url)
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Data from end date for {data.get('symbol')}")
                print(f"   📊 Data points: {len(data.get('data', []))}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
                print(f"   📄 Response Headers: {dict(response.headers)}")
                print(f"   📝 Response Text: {response.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")
        
        print()
        
        # Test 6: Get news data
        print("6️⃣ Testing GET /data/news")
        news_url = f"{BASE_URL}/data/news?symbol=AAPL"
        print(f"   🔗 URL: {news_url}")
        try:
            response = await client.get(news_url)
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! News data for {data.get('symbol')}")
                print(f"   📰 Articles found: {data.get('total_articles', 0)}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
                print(f"   📄 Response Headers: {dict(response.headers)}")
                print(f"   📝 Response Text: {response.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")
        
        print()
        
        # Test 7: Data cleanup
        print("7️⃣ Testing POST /data/cleanup")
        cleanup_url = f"{BASE_URL}/data/cleanup"
        print(f"   🔗 URL: {cleanup_url}")
        try:
            response = await client.post(cleanup_url)
            print(f"   📡 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Cleanup status: {data.get('status')}")
                print(f"   🧹 Files processed: {data.get('files_processed', 0)}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
                print(f"   📄 Response Headers: {dict(response.headers)}")
                print(f"   📝 Response Text: {response.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Error Type: {type(e).__name__}")

        print("\n🎉 Testing completed!")

def run_tests():
    """Run the async tests."""
    print("Starting API endpoint tests...")
    print("Make sure your API server is running on", BASE_URL)
    print("-" * 50)
    
    try:
        asyncio.run(test_endpoints())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")

if __name__ == "__main__":
    run_tests()
