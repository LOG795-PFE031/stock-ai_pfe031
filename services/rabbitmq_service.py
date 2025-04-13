"""
RabbitMQ publisher service for Stock AI.
"""
import pika
import json
from datetime import datetime
import time
from typing import Dict, Any, Optional
import atexit
import threading
import uuid
import socket
import ssl
import os

from core.config import config
from core.logging import logger

class RabbitMQService:
    """Service for publishing messages to RabbitMQ."""
    
    def __init__(self):
        """Initialize RabbitMQ connection and channel."""
        self.exchange_name = 'quote-exchange'  # This matches the C# configuration
        
        # Get host from environment variable if available, otherwise use config
        self.host = os.environ.get('RABBITMQ_HOST', config.rabbitmq.HOST)
        self.port = int(os.environ.get('RABBITMQ_PORT', config.rabbitmq.PORT))
        
        self.connection_params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=pika.PlainCredentials(
                config.rabbitmq.USER,
                config.rabbitmq.PASSWORD
            ),
            heartbeat=30,
            blocked_connection_timeout=10,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=5,
            stack_timeout=5,
            frame_max=131072,  # Set maximum frame size to 128KB
            tcp_options={
                'TCP_KEEPIDLE': 60,
                'TCP_KEEPINTVL': 10,
                'TCP_KEEPCNT': 3
            }
        )
        self.connection = None
        self.channel = None
        self._is_shutting_down = False
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.logger = logger['rabbitmq']
        atexit.register(self._cleanup_on_exit)
        
        # Log configuration (without sensitive data)
        self.logger.info(f"Initializing RabbitMQ service with host={self.host}, port={self.port}")
        self.logger.info(f"Environment RABBITMQ_HOST: {os.environ.get('RABBITMQ_HOST', 'not set')}")
        self.logger.info(f"Environment RABBITMQ_PORT: {os.environ.get('RABBITMQ_PORT', 'not set')}")
        
        # Try to connect, but don't fail initialization if connection fails
        try:
            self.connect()
        except Exception as e:
            self.logger.error(f"Initial connection attempt failed: {str(e)}")
            self.logger.warning("Service will continue in disconnected state. Connection will be retried on first publish attempt.")
    
    def _on_connection_open(self, connection):
        """Called when the connection is opened."""
        self.logger.info("Connection opened")
        self.connection = connection
        self.connection.channel(on_open_callback=self._on_channel_open)
    
    def _on_connection_closed(self, connection, reason):
        """Called when the connection is closed."""
        self.logger.warning(f"Connection closed: {reason}")
        self._cleanup(force=True)
        if not self._is_shutting_down:
            self.connect()
    
    def _on_channel_open(self, channel):
        """Called when the channel is opened."""
        self.logger.info("Channel opened")
        self.channel = channel
        self.channel.exchange_declare(
            exchange=self.exchange_name,
            exchange_type='fanout',
            durable=True,
            callback=self._on_exchange_declared
        )
    
    def _on_channel_closed(self, channel, reason):
        """Called when the channel is closed."""
        self.logger.warning(f"Channel closed: {reason}")
        if self.connection and not self.connection.is_closed:
            self.connection.close()
    
    def _on_exchange_declared(self, frame):
        """Called when the exchange declare is confirmed."""
        self.logger.info(f"Exchange {self.exchange_name} declared")
        self._event.set()
    
    def connect(self):
        """Establish connection to RabbitMQ with retries."""
        if self._is_shutting_down:
            return
        
        with self._lock:
            for attempt in range(5):  # Max 5 retries
                try:
                    self.logger.info(f"Attempting to connect to RabbitMQ (attempt {attempt + 1}/5)")
                    self.logger.debug(f"Connection parameters: host={self.host}, port={self.port}")
                    
                    # Test TCP connection first
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        sock.connect((self.host, self.port))
                        sock.close()
                        self.logger.info("TCP connection test successful")
                    except socket.error as e:
                        self.logger.error(f"TCP connection failed: {str(e)}")
                        self.logger.error(f"Host: {self.host}, Port: {self.port}")
                        self.logger.error("Please check if RabbitMQ is running and accessible")
                        if attempt == 4:  # Last attempt
                            raise
                        time.sleep(5)  # Wait before retrying
                        continue
                    
                    self._event.clear()
                    self.connection = pika.SelectConnection(
                        parameters=self.connection_params,
                        on_open_callback=self._on_connection_open,
                        on_open_error_callback=lambda conn, err: self.logger.error(f"Connection open failed: {str(err)}"),
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
                        self.logger.info("Successfully connected to RabbitMQ")
                        return
                    else:
                        self.logger.warning("Timeout waiting for connection setup")
                        self._cleanup(force=True)
                        
                except Exception as e:
                    if attempt == 4:  # Last attempt
                        self.logger.error(f"Failed to connect to RabbitMQ after 5 attempts: {str(e)}")
                        self.logger.error("Please check the following:")
                        self.logger.error("1. Is RabbitMQ running?")
                        self.logger.error("2. Are the host and port correct?")
                        self.logger.error("3. Is the RabbitMQ server accessible from this container?")
                        self.logger.error("4. Are the credentials correct?")
                        raise
                    self.logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(5)  # Wait 5 seconds before retrying
    
    def _run_io_loop(self):
        """Run the IO loop in a separate thread."""
        try:
            self.connection.ioloop.start()
        except Exception as e:
            self.logger.error(f"IO loop terminated with error: {str(e)}")
        finally:
            self.logger.info("IO loop stopped")
    
    def publish_stock_quote(self, symbol: str, prediction: Dict[str, Any]) -> bool:
        """
        Publish a stock quote to RabbitMQ.
        
        Args:
            symbol: Stock symbol
            prediction: Prediction data dictionary
            
        Returns:
            bool: True if message was published successfully, False otherwise
        """
        if self._is_shutting_down:
            self.logger.warning("Cannot publish during shutdown")
            return False
        
        with self._lock:
            for attempt in range(3):  # Max 3 retries
                try:
                    # Ensure connection and channel are open
                    if not self.connection or self.connection.is_closed or not self.channel or self.channel.is_closed:
                        try:
                            self.connect()
                            if not self._event.wait(timeout=10):
                                raise pika.exceptions.AMQPConnectionError("Failed to establish connection")
                        except Exception as e:
                            self.logger.error(f"Failed to establish connection for publishing: {str(e)}")
                            return False
                    
                    # Generate a unique message ID
                    message_id = str(uuid.uuid4())
                    timestamp = datetime.utcnow()
                    
                    # Structure message to match C# StockQuote type
                    message = {
                        "Symbol": symbol,
                        "Price": float(prediction['prediction']),  # Convert to float to match C# decimal
                        "Date": timestamp.isoformat(),  # ISO format for DateTime
                        "CorrelationId": message_id,  # Add correlation ID for tracking
                        "MessageType": "StockQuote"  # Add message type for MassTransit
                    }
                    
                    # Serialize message
                    message_body = json.dumps(message).encode('utf-8')
                    
                    # Check message size
                    if len(message_body) > self.connection_params.frame_max:
                        self.logger.error(f"Message size ({len(message_body)} bytes) exceeds maximum frame size")
                        return False
                    
                    # Create properties
                    properties = pika.BasicProperties(
                        delivery_mode=2,  # Persistent message
                        content_type='application/json',
                        content_encoding='utf-8',
                        message_id=message_id,
                        correlation_id=message_id,
                        timestamp=int(time.time()),
                        type='StockQuote',  # Match C# message type
                        headers={
                            'MT-Activity-Id': message_id,
                            'MT-Message-Type': 'StockQuote'
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
                    
                    self.logger.info(f"Successfully published stock quote for {symbol}")
                    return True
                    
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        self.logger.error(f"Failed to publish message for {symbol} after 3 attempts: {str(e)}")
                        return False
                    self.logger.warning(f"Publish attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2)  # Wait 2 seconds before retrying
    
    def _cleanup(self, force: bool = False):
        """Clean up RabbitMQ resources."""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
        except Exception as e:
            if force:
                self.logger.error(f"Error during forced cleanup: {str(e)}")
            else:
                self.logger.warning(f"Error during cleanup: {str(e)}")
    
    def _cleanup_on_exit(self):
        """Clean up on application exit."""
        self._is_shutting_down = True
        self._cleanup(force=True)
    
    def close(self):
        """Close the RabbitMQ connection."""
        self._is_shutting_down = True
        self._cleanup(force=True) 