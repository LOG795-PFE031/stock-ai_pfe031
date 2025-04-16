"""
Script to monitor prediction events from RabbitMQ.
"""
import pika
import json
import os
import socket
import time
import logging
from pika.exceptions import AMQPConnectionError, AMQPHeartbeatTimeout

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

logger.info(f"🔌 Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")

def callback(ch, method, properties, body):
    """Callback function to handle incoming messages."""
    try:
        message = json.loads(body)
        logger.info("\n📊 Received prediction:")
        logger.info(f"Symbol: {message.get('Symbol')}")
        logger.info(f"Price: ${message.get('Price'):.2f}")
        logger.info(f"Date: {message.get('Date')}")
        logger.info(f"Model Type: {message.get('ModelType')}")
        logger.info(f"Confidence: {message.get('Confidence'):.1%}")
        logger.info(f"Model Version: {message.get('ModelVersion')}")
        logger.info(f"Correlation ID: {message.get('CorrelationId')}")
        logger.info("-" * 50)
    except json.JSONDecodeError:
        logger.error(f"❌ Received non-JSON message: {body}")
    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")

def create_connection():
    """Create a new RabbitMQ connection."""
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=60,  # Increased heartbeat interval
            blocked_connection_timeout=30,
            connection_attempts=5,
            retry_delay=5
        )
        connection = pika.BlockingConnection(parameters)
        logger.info("✅ Successfully connected to RabbitMQ")
        return connection
    except Exception as e:
        logger.error(f"❌ Failed to create connection: {str(e)}")
        raise

def monitor_predictions():
    """Monitor prediction events from RabbitMQ."""
    max_retries = 5
    retry_count = 0
    retry_delay = 5  # seconds
    
    while retry_count < max_retries:
        try:
            # Create connection
            connection = create_connection()
            channel = connection.channel()
            
            # Declare the exchange
            channel.exchange_declare(
                exchange='quote-exchange',
                exchange_type='fanout',
                durable=True
            )
            logger.info("✅ Successfully declared quote-exchange")
            
            # Create a temporary queue
            result = channel.queue_declare(queue='', exclusive=True)
            queue_name = result.method.queue
            logger.info(f"✅ Created temporary queue: {queue_name}")
            
            # Bind the queue to the exchange
            channel.queue_bind(
                exchange='quote-exchange',
                queue=queue_name
            )
            logger.info("✅ Successfully bound queue to exchange")
            
            logger.info("👂 Listening for prediction events...")
            logger.info("Press Ctrl+C to stop")
            
            # Start consuming
            channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=True
            )
            
            channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("\n👋 Stopping monitor...")
            if 'connection' in locals() and connection.is_open:
                connection.close()
            break
        except (AMQPConnectionError, AMQPHeartbeatTimeout) as e:
            retry_count += 1
            logger.error(f"❌ Connection error (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("❌ Max retries reached. Exiting...")
                break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            if 'connection' in locals() and connection.is_open:
                connection.close()
            break

if __name__ == "__main__":
    monitor_predictions() 