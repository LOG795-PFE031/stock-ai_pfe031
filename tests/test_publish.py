import pika
import json
from datetime import datetime
import uuid
import time
import socket

def test_connection(host, port):
    """Test TCP connection to RabbitMQ."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error testing connection: {str(e)}")
        return False

# Test connection first
print("Testing connection to RabbitMQ...")
if not test_connection('localhost', 5672):
    print("❌ Could not establish TCP connection to RabbitMQ")
    print("Please check that:")
    print("1. RabbitMQ container is running")
    print("2. Port 5672 is accessible")
    print("3. No firewall is blocking the connection")
    exit(1)
print("✅ TCP connection test successful")

# Connection parameters
connection_params = pika.ConnectionParameters(
    host='localhost',
    port=5672,
    credentials=pika.PlainCredentials('guest', 'guest'),
    heartbeat=30,
    blocked_connection_timeout=30,
    connection_attempts=5,
    retry_delay=5,
    socket_timeout=30,
    stack_timeout=30,
    frame_max=131072,
    tcp_options={
        'TCP_KEEPIDLE': 60,
        'TCP_KEEPINTVL': 10,
        'TCP_KEEPCNT': 3
    }
)

try:
    print("Attempting to connect to RabbitMQ...")
    connection = pika.BlockingConnection(connection_params)
    print("✅ Successfully connected to RabbitMQ")
    
    channel = connection.channel()
    print("✅ Successfully opened channel")
    
    # Declare exchange
    print("Declaring quote-exchange...")
    channel.exchange_declare(
        exchange='quote-exchange',
        exchange_type='fanout',
        durable=True
    )
    print("✅ Successfully declared quote-exchange")
    
    # Create test message
    message_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    message = {
        "Symbol": "TEST",
        "Price": 100.0,
        "Date": timestamp.isoformat(),
        "CorrelationId": message_id,
        "MessageType": "StockQuote",
        "ModelType": "test",
        "Confidence": 0.95,
        "ModelVersion": "1.0.0"
    }
    
    # Create properties
    properties = pika.BasicProperties(
        delivery_mode=2,  # Persistent message
        content_type='application/json',
        content_encoding='utf-8',
        message_id=message_id,
        correlation_id=message_id,
        timestamp=int(time.time()),
        type='StockQuote',
        headers={
            'MT-Activity-Id': message_id,
            'MT-Message-Type': 'StockQuote',
            'MT-Model-Type': 'test'
        }
    )
    
    # Publish message
    print(f"Publishing test message to quote-exchange: {json.dumps(message, indent=2)}")
    channel.basic_publish(
        exchange='quote-exchange',
        routing_key='',
        body=json.dumps(message).encode('utf-8'),
        properties=properties
    )
    
    print("✅ Message published successfully!")
    
except pika.exceptions.AMQPConnectionError as e:
    print(f"❌ Failed to connect to RabbitMQ: {str(e)}")
    print("Please check that:")
    print("1. RabbitMQ is running and accessible")
    print("2. The credentials are correct")
    print("3. The port is correct")
    exit(1)
except pika.exceptions.AMQPChannelError as e:
    print(f"❌ Channel error: {str(e)}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")
    exit(1)
finally:
    try:
        if 'connection' in locals() and connection.is_open:
            connection.close()
            print("✅ Connection closed")
    except Exception as e:
        print(f"❌ Error closing connection: {str(e)}") 