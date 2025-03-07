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
QUEUE_NAME = 'prediction_queue_2'  # Using the second queue
ROUTING_KEY = 'google.stock.prediction'  # Same routing key as producer

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
        
        # Ensure the exchange exists
        channel.exchange_declare(
            exchange=EXCHANGE_NAME,
            exchange_type='topic',
            durable=True
        )
        
        # Ensure the queue exists and is bound to the exchange
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.queue_bind(
            exchange=EXCHANGE_NAME,
            queue=QUEUE_NAME,
            routing_key=ROUTING_KEY
        )
        
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
        
        # More detailed logging in this consumer
        logger.info("=" * 50)
        logger.info(f"ALERT: New stock prediction received from {model} model")
        logger.info(f"Stock: {ticker}")
        logger.info(f"Date: {date}")
        logger.info(f"Predicted Opening Price: ${price:.2f}")
        
        # Simulate doing some additional processing
        logger.info("Performing additional analysis...")
        time.sleep(1)  # Simulate work
        
        # Calculate a simple 5% price range
        price_low = price * 0.95
        price_high = price * 1.05
        logger.info(f"Predicted price range: ${price_low:.2f} - ${price_high:.2f}")
        logger.info("=" * 50)
        
        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except json.JSONDecodeError:
        logger.error(f"Failed to parse message as JSON: {body}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

def consume_predictions():
    """Start consuming predictions from RabbitMQ."""
    connection = None
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
        
        logger.info(f"Advanced consumer started. Waiting for messages on queue: {QUEUE_NAME}...")
        
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