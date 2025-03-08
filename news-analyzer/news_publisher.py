import pika
import json
import os
from datetime import datetime
import logging
import time
import uuid
import threading
import atexit

logger = logging.getLogger(__name__)

class NewsPublisher:
    def __init__(self, host='rabbitmq', exchange='news-exchange', exchange_type='fanout',
                 max_retries=5, retry_delay=5):
        """Initialize the RabbitMQ connection for news publishing.

        Args:
            host (str): RabbitMQ host (default from env or 'rabbitmq').
            exchange (str): Exchange name ('news-exchange').
            exchange_type (str): Exchange type ('fanout').
            max_retries (int): Maximum connection retry attempts.
            retry_delay (int): Delay between retry attempts in seconds.
        """
        self.host = os.environ.get('RABBITMQ_HOST', host)
        self.exchange = exchange
        self.exchange_type = exchange_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection = None
        self.channel = None
        self._is_shutting_down = False
        self._lock = threading.Lock()
        atexit.register(self._cleanup_on_exit)
        self.connect()

    def connect(self):
        """Establish connection to RabbitMQ server with retries."""
        if self._is_shutting_down:
            return False

        with self._lock:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Connecting to RabbitMQ (attempt {attempt + 1}/{self.max_retries})")
                    credentials = pika.PlainCredentials(
                        os.environ.get('RABBITMQ_USERNAME', 'guest'),
                        os.environ.get('RABBITMQ_PASSWORD', 'guest')
                    )
                    parameters = pika.ConnectionParameters(
                        host=self.host,
                        port=int(os.environ.get('RABBITMQ_PORT', 5672)),
                        credentials=credentials,
                        heartbeat=30,
                        blocked_connection_timeout=10,
                        connection_attempts=3,
                        retry_delay=2,
                        socket_timeout=5
                    )
                    self.connection = pika.BlockingConnection(parameters)
                    self.channel = self.connection.channel()
                    self.channel.confirm_delivery()
                    self.channel.exchange_declare(
                        exchange=self.exchange,
                        exchange_type=self.exchange_type,
                        durable=True
                    )
                    logger.info(f"Connected to RabbitMQ at {self.host}, exchange: {self.exchange}")
                    return True
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to connect after {self.max_retries} attempts: {str(e)}")
                        return False
                    logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(self.retry_delay)
            return False

    def publish_news(self, title, symbol, content, published_at=None, opinion=0):
        """Publish news article to RabbitMQ in the expected format.

        Args:
            title (str): Article title.
            symbol (str): Stock ticker.
            content (str): Article text.
            published_at (datetime, optional): Article publication date.
            opinion (int, optional): Sentiment score (-1, 0, 1).

        Returns:
            bool: True if published successfully, False otherwise.
        """
        if self._is_shutting_down:
            logger.warning("Cannot publish during shutdown")
            return False

        with self._lock:
            max_publish_retries = 3
            for attempt in range(max_publish_retries):
                try:
                    if not self.connection or self.connection.is_closed or not self.channel or self.channel.is_closed:
                        logger.info(f"Connection closed, reconnecting...")
                        if not self.connect():
                            raise Exception("Failed to establish connection")

                    message_id = str(uuid.uuid4())
                    timestamp = datetime.utcnow().isoformat(timespec='microseconds') + 'Z'

                    message = {
                        "messageId": message_id,
                        "conversationId": message_id,
                        "messageType": ["urn:message:Application.Commands.News:AddNews"],
                        "message": {
                            "Id": message_id,
                            "Title": title,
                            "Symbol": symbol,
                            "Content": content[:1000] + "..." if len(content) > 1000 else content,  # Truncate long content for logging
                            "PublishedAt": (published_at.isoformat() if published_at 
                                           else datetime.utcnow().isoformat()),
                            "Opinion": opinion
                        },
                        "sentTime": timestamp,
                        "headers": {
                            "MT-Activity-Id": message_id,
                            "MT-Message-Type": "AddNews"
                        }
                    }

                    # Log the message format before publishing (truncated for readability)
                    log_message = {k: v for k, v in message.items() if k != "message"}
                    log_message["message"] = {
                        "Id": message["message"]["Id"],
                        "Title": message["message"]["Title"],
                        "Symbol": message["message"]["Symbol"],
                        "PublishedAt": message["message"]["PublishedAt"],
                        "Opinion": message["message"]["Opinion"],
                        "Content": message["message"]["Content"][:100] + "..." if len(message["message"]["Content"]) > 100 else message["message"]["Content"]
                    }
                    logger.info(f"Publishing message: {json.dumps(log_message)}")
                    logger.info(f"Publishing to exchange: {self.exchange}, type: {self.exchange_type}")

                    message_body = json.dumps(message).encode('utf-8')
                    properties = pika.BasicProperties(
                        delivery_mode=2,  # Persistent
                        content_type='application/vnd.masstransit+json',
                        content_encoding='utf-8',
                        message_id=message_id,
                        correlation_id=message_id,
                        timestamp=int(time.time()),
                        type='urn:message:Application.Commands.News:AddNews',
                        headers={
                            'MT-Activity-Id': message_id,
                            'MT-Message-Type': 'AddNews'
                        }
                    )

                    # Check if exchange exists and has bindings
                    try:
                        exchange_info = self.channel.exchange_declare(
                            exchange=self.exchange,
                            exchange_type=self.exchange_type,
                            durable=True,
                            passive=True  # Just check if it exists
                        )
                        logger.info(f"Exchange {self.exchange} exists")
                    except Exception as e:
                        logger.warning(f"Exchange check failed: {str(e)}")
                    
                    # Try to publish with confirmation
                    self.channel.basic_publish(
                        exchange=self.exchange,
                        routing_key='',  # Fanout exchange doesn't need routing key
                        body=message_body,
                        properties=properties,
                        mandatory=True
                    )

                    logger.info(f"Published news for {symbol}: {title}")
                    return True
                except pika.exceptions.UnroutableError:
                    logger.error(f"Message for {symbol} returned: No queue bound to exchange {self.exchange}")
                    return False
                except Exception as e:
                    if attempt == max_publish_retries - 1:
                        logger.error(f"Failed to publish {symbol} after {max_publish_retries} attempts: {str(e)}")
                        return False
                    logger.warning(f"Publish attempt {attempt + 1} failed: {str(e)}, retrying...")
                    self._cleanup(force=True)
                    time.sleep(self.retry_delay)
            return False

    def _cleanup(self, force=False):
        """Clean up connection and channel."""
        with self._lock:
            if self.channel and (force or not self.channel.is_closed):
                try:
                    self.channel.close()
                except Exception as e:
                    logger.debug(f"Error closing channel: {str(e)}")
            if self.connection and (force or not self.connection.is_closed):
                try:
                    self.connection.close()
                except Exception as e:
                    logger.debug(f"Error closing connection: {str(e)}")
            self.channel = None
            self.connection = None

    def _cleanup_on_exit(self):
        """Cleanup on program exit."""
        self._is_shutting_down = True
        self._cleanup(force=True)
        logger.info("NewsPublisher cleaned up on exit")

    def close(self):
        """Close the connection."""
        self._is_shutting_down = True
        self._cleanup()
        logger.info("Closed RabbitMQ connection")