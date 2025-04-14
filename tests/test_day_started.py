"""
Script to manually test the day-started event handling.
"""
import pika
import json
from datetime import datetime
import time
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

def publish_day_started():
    """Publish a day-started event to RabbitMQ."""
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
            exchange='day-started-exchange',
            exchange_type='fanout',
            durable=True
        )
        
        # Create the message
        message = {
            "event_type": "DayStarted",
            "timestamp": datetime.now().isoformat(),
            "day": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Publish the message
        channel.basic_publish(
            exchange='day-started-exchange',
            routing_key='',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/json'
            )
        )
        
        print(f"✅ Published day-started event: {json.dumps(message, indent=2)}")
        
        # Close the connection
        connection.close()
        
    except Exception as e:
        print(f"❌ Error publishing day-started event: {str(e)}")
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    print("🔄 Publishing day-started event...")
    publish_day_started() 