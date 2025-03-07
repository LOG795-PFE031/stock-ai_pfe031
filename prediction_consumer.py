import pika
import json
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RabbitMQ configuration
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'guest'
RABBITMQ_PASS = 'guest'
EXCHANGE_NAME = 'stock_predictions'
QUEUE_NAME = 'prediction_queue_1'  # Change to 'prediction_queue_2' to test the other queue

def connect_rabbitmq():
    """Establish connection to RabbitMQ server."""
    try:
        # Connect to RabbitMQ
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT, 
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        logger.info(f"Connected to RabbitMQ: {RABBITMQ_HOST}:{RABBITMQ_PORT}")
        return connection, channel
    
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {e}")
        raise

def callback(ch, method, properties, body):
    """Process messages from RabbitMQ."""
    try:
        # Parse JSON message
        message = json.loads(body)
        
        # Log the received prediction
        ticker = message.get('ticker', 'Unknown')
        date = message.get('prediction_date', 'Unknown')
        price = message.get('predicted_open', 0.0)
        model = message.get('model', 'Unknown')
        
        logger.info(f"Consumer {QUEUE_NAME} received prediction:")
        logger.info(f"  Ticker: {ticker}")
        logger.info(f"  Date: {date}")
        logger.info(f"  Predicted Open: ${price:.2f}")
        logger.info(f"  Model: {model}")
        
        # Here you would typically do something with the prediction
        # For example, store it in a database, trigger a trade, etc.
        
        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except json.JSONDecodeError:
        logger.error(f"Failed to parse message as JSON: {body}")
        # Still acknowledge to remove from queue
        ch.basic_ack(delivery_tag=method.delivery_tag)
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # In case of error, we still acknowledge to avoid message buildup
        # In a real system, you might want to use negative acknowledgment instead
        ch.basic_ack(delivery_tag=method.delivery_tag)

def consume_predictions():
    """Start consuming predictions from RabbitMQ."""
    try:
        # Connect to RabbitMQ
        connection, channel = connect_rabbitmq()
        
        # Set prefetch count to 1 to ensure fair dispatch
        channel.basic_qos(prefetch_count=1)
        
        # Set up the consumer
        channel.basic_consume(
            queue=QUEUE_NAME,
            on_message_callback=callback
        )
        
        logger.info(f"Consumer started. Waiting for messages on queue: {QUEUE_NAME}...")
        
        # Start consuming messages
        channel.start_consuming()
    
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
        if connection and connection.is_open:
            connection.close()
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error in consumer: {e}")
        if connection and connection.is_open:
            connection.close()
        sys.exit(1)

if __name__ == "__main__":
    try:
        consume_predictions()
    except Exception as e:
        logger.error(f"Failed to start consumer: {e}")
        sys.exit(1)