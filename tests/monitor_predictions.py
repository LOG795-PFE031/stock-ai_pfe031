"""
Script to monitor prediction events from RabbitMQ.
"""
import pika
import json
import os
import socket

def get_rabbitmq_host():
    """Get the RabbitMQ host based on the environment."""
    # Try to resolve the hostname
    try:
        socket.gethostbyname('rabbitmq')
        return 'rabbitmq'  # If we can resolve it, we're in Docker
    except socket.gaierror:
        return 'localhost'  # If not, we're running locally

# RabbitMQ connection parameters
RABBITMQ_HOST = get_rabbitmq_host()
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.environ.get('RABBITMQ_PASS', 'guest')

print(f"🔌 Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")

def callback(ch, method, properties, body):
    """Callback function to handle incoming messages."""
    try:
        message = json.loads(body)
        print("\n📊 Received prediction:")
        print(f"Symbol: {message.get('Symbol')}")
        print(f"Price: ${message.get('Price'):.2f}")
        print(f"Date: {message.get('Date')}")
        print(f"Model Type: {message.get('MessageType')}")
        print(f"Correlation ID: {message.get('CorrelationId')}")
        print("-" * 50)
    except json.JSONDecodeError:
        print(f"❌ Received non-JSON message: {body}")

def monitor_predictions():
    """Monitor prediction events from RabbitMQ."""
    try:
        # Create connection parameters
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=30,
            blocked_connection_timeout=10,
            connection_attempts=3,
            retry_delay=2
        )
        
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # Declare the exchange
        channel.exchange_declare(
            exchange='quote-exchange',
            exchange_type='fanout',
            durable=True
        )
        
        # Create a temporary queue
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        
        # Bind the queue to the exchange
        channel.queue_bind(
            exchange='quote-exchange',
            queue=queue_name
        )
        
        print("👂 Listening for prediction events...")
        print("Press Ctrl+C to stop")
        
        # Start consuming
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback,
            auto_ack=True
        )
        
        channel.start_consuming()
        
    except KeyboardInterrupt:
        print("\n👋 Stopping monitor...")
        if 'connection' in locals():
            connection.close()
    except Exception as e:
        print(f"❌ Error monitoring predictions: {str(e)}")
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    monitor_predictions() 