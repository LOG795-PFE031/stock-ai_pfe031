import pika
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def callback(ch, method, properties, body):
    """Handle received messages"""
    try:
        message = json.loads(body)
        logger.info(f"Received stock quote: {json.dumps(message, indent=2)}")
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")

def main():
    # Connection parameters
    connection_params = pika.ConnectionParameters(
        host='localhost',  # or your RabbitMQ host
        port=5672,
        credentials=pika.PlainCredentials('guest', 'guest')
    )

    try:
        # Establish connection
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()

        # Declare the exchange (should match the publisher)
        exchange_name = 'quote-exchange'
        channel.exchange_declare(
            exchange=exchange_name,
            exchange_type='fanout',
            durable=True
        )

        # Create a temporary queue
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        # Bind to the exchange (no routing key needed for fanout)
        channel.queue_bind(
            exchange=exchange_name,
            queue=queue_name
        )

        logger.info(f"Waiting for stock quotes. To exit press CTRL+C")

        # Start consuming messages
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback,
            auto_ack=True
        )

        channel.start_consuming()

    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
        if connection and not connection.is_closed:
            connection.close()
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main() 