import pika
import json
from datetime import datetime
import logging
import time
from typing import Dict, Any, Optional, Callable
import atexit
import threading
import uuid
from collections import deque

logger = logging.getLogger(__name__)

class RabbitMQPublisher:
    def __init__(self, host: str = 'rabbitmq', port: int = 5672, 
                 username: str = 'guest', password: str = 'guest',
                 max_retries: int = 5, retry_delay: int = 5):
        """Initialize RabbitMQ connection and channel"""
        self.exchange_name = 'quote-exchange'
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=pika.PlainCredentials(username, password),
            heartbeat=30,
            blocked_connection_timeout=10,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=5,
            stack_timeout=5,
            frame_max=131072,  # Set maximum frame size to 128KB
            tcp_options={'TCP_KEEPIDLE': 60,
                        'TCP_KEEPINTVL': 10,
                        'TCP_KEEPCNT': 3}
        )
        self.connection = None
        self.channel = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._is_shutting_down = False
        self._lock = threading.Lock()
        self._delivery_confirmations = {}
        self._message_number = 0
        self._event = threading.Event()
        self._current_message = None
        self._current_result = None
        atexit.register(self._cleanup_on_exit)
        self.connect()

    def _on_connection_open(self, connection):
        """Called when the connection is opened"""
        logger.info("Connection opened")
        self.connection = connection
        self.connection.channel(on_open_callback=self._on_channel_open)

    def _on_connection_closed(self, connection, reason):
        """Called when the connection is closed"""
        logger.warning(f"Connection closed: {reason}")
        self._cleanup(force=True)
        if not self._is_shutting_down:
            self.connect()

    def _on_channel_open(self, channel):
        """Called when the channel is opened"""
        logger.info("Channel opened")
        self.channel = channel
        self.channel.confirm_delivery(self._on_delivery_confirmation)
        self.channel.exchange_declare(
            exchange=self.exchange_name,
            exchange_type='fanout',
            durable=True,
            callback=self._on_exchange_declared
        )

    def _on_channel_closed(self, channel, reason):
        """Called when the channel is closed"""
        logger.warning(f"Channel closed: {reason}")
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def _on_exchange_declared(self, frame):
        """Called when the exchange declare is confirmed"""
        logger.info(f"Exchange {self.exchange_name} declared")
        self._event.set()

    def _on_delivery_confirmation(self, method_frame):
        """Called when RabbitMQ confirms message delivery"""
        confirmation_type = method_frame.method.NAME.split('.')[1].lower()
        delivery_tag = method_frame.method.delivery_tag
        
        if delivery_tag in self._delivery_confirmations:
            if confirmation_type == 'ack':
                self._delivery_confirmations[delivery_tag] = True
            elif confirmation_type == 'nack':
                self._delivery_confirmations[delivery_tag] = False
            
            if delivery_tag == self._message_number:
                self._current_result = self._delivery_confirmations[delivery_tag]
                self._event.set()

    def connect(self):
        """Establish connection to RabbitMQ with retries"""
        if self._is_shutting_down:
            return

        with self._lock:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Attempting to connect to RabbitMQ (attempt {attempt + 1}/{self.max_retries})")
                    self._event.clear()
                    self.connection = pika.SelectConnection(
                        parameters=self.connection_params,
                        on_open_callback=self._on_connection_open,
                        on_open_error_callback=lambda conn, err: logger.error(f"Connection open failed: {err}"),
                        on_close_callback=self._on_connection_closed
                    )
                    
                    # Start the IO loop in a separate thread
                    self._io_thread = threading.Thread(
                        target=self._run_io_loop,
                        name="RabbitMQ-IOLoop",
                        daemon=True
                    )
                    self._io_thread.start()
                    
                    # Wait for exchange declaration
                    if self._event.wait(timeout=10):
                        logger.info("Successfully connected to RabbitMQ")
                        return
                    else:
                        logger.warning("Timeout waiting for connection setup")
                        self._cleanup(force=True)
                        
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to connect to RabbitMQ after {self.max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(self.retry_delay)

    def _run_io_loop(self):
        """Run the IO loop in a separate thread"""
        try:
            self.connection.ioloop.start()
        except Exception as e:
            logger.error(f"IO loop terminated with error: {str(e)}")
        finally:
            logger.info("IO loop stopped")

    def publish_stock_quote(self, symbol: str, prediction: Dict[str, Any]) -> bool:
        """
        Publish a stock quote to RabbitMQ with guaranteed delivery
        Returns True if message was confirmed delivered, False otherwise
        """
        if self._is_shutting_down:
            logger.warning("Cannot publish during shutdown")
            return False

        with self._lock:
            max_publish_retries = 3
            for attempt in range(max_publish_retries):
                try:
                    # Ensure connection and channel are open
                    if not self.connection or self.connection.is_closed or not self.channel or self.channel.is_closed:
                        self.connect()
                        if not self._event.wait(timeout=10):
                            raise pika.exceptions.AMQPConnectionError("Failed to establish connection")

                    # Generate a unique message ID
                    message_id = str(uuid.uuid4())
                    timestamp = datetime.utcnow().isoformat(timespec='microseconds') + 'Z'

                    # Structure message
                    message = {
                        "messageId": message_id,
                        "conversationId": message_id,
                        "messageType": ["urn:message:Application.Commands.Quotes:AddQuote"],
                        "message": {
                            "Id": message_id,
                            "Symbol": symbol,
                            "Price": prediction['prediction'],
                            "Timestamp": timestamp,
                            "Metadata": {
                                "ConfidenceScore": prediction['confidence_score'],
                                "ModelVersion": prediction['model_version'],
                                "ModelType": prediction['model_type']
                            }
                        },
                        "sentTime": timestamp,
                        "headers": {
                            "MT-Activity-Id": message_id,
                            "MT-Message-Type": "AddQuote"
                        }
                    }

                    # Serialize message
                    message_body = json.dumps(message).encode('utf-8')
                    
                    # Check message size
                    if len(message_body) > self.connection_params.frame_max:
                        logger.error(f"❌ Message size ({len(message_body)} bytes) exceeds maximum frame size")
                        return False

                    # Prepare for delivery confirmation
                    self._message_number += 1
                    self._delivery_confirmations[self._message_number] = None
                    self._current_message = self._message_number
                    self._current_result = None
                    self._event.clear()

                    # Create properties
                    properties = pika.BasicProperties(
                        delivery_mode=2,
                        content_type='application/vnd.masstransit+json',
                        content_encoding='utf-8',
                        message_id=message_id,
                        correlation_id=message_id,
                        timestamp=int(time.time()),
                        type='urn:message:Application.Commands.Quotes:AddQuote',
                        headers={
                            'MT-Activity-Id': message_id,
                            'MT-Message-Type': 'AddQuote'
                        }
                    )

                    # Publish message
                    self.channel.basic_publish(
                        exchange=self.exchange_name,
                        routing_key='',
                        body=message_body,
                        properties=properties,
                        mandatory=True
                    )

                    # Wait for confirmation
                    if self._event.wait(timeout=5):
                        if self._current_result:
                            logger.info(f"✅ Successfully published and confirmed stock quote for {symbol}")
                            return True
                        else:
                            logger.warning(f"⚠️ Message for {symbol} was not confirmed")
                            raise pika.exceptions.UnroutableError()
                    else:
                        logger.warning(f"⚠️ Timed out waiting for confirmation of message for {symbol}")
                        raise pika.exceptions.UnroutableError()

                except pika.exceptions.UnroutableError:
                    logger.error(f"❌ Message for {symbol} was returned: No queue bound to exchange")
                    return False
                except Exception as e:
                    if attempt == max_publish_retries - 1:
                        logger.error(f"❌ Failed to publish message for {symbol} after {max_publish_retries} attempts: {str(e)}")
                        return False
                    logger.warning(f"Publish attempt {attempt + 1} failed: {str(e)}, retrying...")
                    self._cleanup(force=True)
                    time.sleep(self.retry_delay)
            
            return False

    def _cleanup(self, force: bool = False):
        """Clean up connection and channel"""
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
            self._delivery_confirmations.clear()
            self._message_number = 0
            self._current_message = None
            self._current_result = None
            self._event.clear()

    def _cleanup_on_exit(self):
        """Cleanup handler registered with atexit"""
        self._is_shutting_down = True
        self._cleanup(force=True)
        if hasattr(self, '_io_thread') and self._io_thread and self._io_thread.is_alive():
            try:
                self._io_thread.join(timeout=5)
            except Exception as e:
                logger.debug(f"Error joining IO thread: {str(e)}")
        logger.info("RabbitMQ publisher cleaned up on exit")

    def close(self):
        """Close the connection"""
        self._is_shutting_down = True
        self._cleanup()
        if hasattr(self, '_io_thread') and self._io_thread and self._io_thread.is_alive():
            try:
                self._io_thread.join(timeout=5)
            except Exception as e:
                logger.debug(f"Error joining IO thread: {str(e)}")
        logger.info("Closed RabbitMQ connection")

# Create a singleton instance
rabbitmq_publisher = RabbitMQPublisher() 