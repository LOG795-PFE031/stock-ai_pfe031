"""
RabbitMQ publisher service for Stock AI.
"""
import pika
import json
from datetime import datetime
import time
from typing import Dict, Any, Optional, Callable
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
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RabbitMQService, cls).__new__(cls)
                cls._instance.logger = logger['rabbitmq']
                cls._instance.logger.debug("✨ Creating new RabbitMQ service instance")
            else:
                cls._instance.logger.debug("✨ Returning existing RabbitMQ service instance")
            return cls._instance
    
    def __init__(self):
        """Initialize RabbitMQ connection and channel."""
        # Skip initialization if already initialized
        if self._initialized:
            self.logger.debug("✨ Skipping initialization - already initialized")
            return
            
        self._initialized = True
        self.logger.debug("✨ Starting RabbitMQ service initialization")
        
        self.exchange_name = 'quote-exchange'  # This matches the C# configuration
        self.day_started_exchange = 'day-started-exchange'
        self.day_started_queue = 'stock-ai-day-started'
        
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
        self.connection = None
        self.channel = None
        self._is_shutting_down = False
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._day_started_callback = None
        self._io_thread = None
        self._connection_state = 'disconnected'
        atexit.register(self._cleanup_on_exit)
        
        # Log configuration (without sensitive data)
        self.logger.info(f"✨ Initializing RabbitMQ service with host={self.host}, port={self.port}")
        self.logger.info(f"✨ Environment RABBITMQ_HOST: {os.environ.get('RABBITMQ_HOST', 'not set')}")
        self.logger.info(f"✨ Environment RABBITMQ_PORT: {os.environ.get('RABBITMQ_PORT', 'not set')}")
        
        # Try to connect, but don't fail initialization if connection fails
        try:
            self.connect()
        except Exception as e:
            self.logger.error(f"❌ Initial connection attempt failed: {str(e)}")
            self.logger.warning("⚠️ Service will continue in disconnected state. Connection will be retried on first publish attempt.")
    
    def set_day_started_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set the callback function for day-started events."""
        self._day_started_callback = callback
        self.logger.info("✅ Day-started callback set")
        
        # If we're already connected, ensure the callback is properly set up
        if self.connection and not self.connection.is_closed and self.channel and not self.channel.is_closed:
            self._setup_day_started_consumer()
    
    def _setup_day_started_consumer(self) -> None:
        """Set up the day-started event consumer."""
        try:
            if not self._day_started_callback:
                self.logger.warning("No day-started callback set, skipping consumer setup")
                return
            
            # Declare queue for day-started events
            self.channel.queue_declare(
                queue=self.day_started_queue,
                durable=True,
                exclusive=False,
                auto_delete=False,
                callback=self._on_day_started_queue_declared
            )
        except Exception as e:
            self.logger.error(f"Error setting up day-started consumer: {str(e)}")
    
    def _on_connection_open(self, connection):
        """Called when the connection is opened."""
        self.logger.info("✨ Connection opened")
        self.connection = connection
        self._connection_state = 'connected'
        self.connection.channel(on_open_callback=self._on_channel_open)
    
    def _on_connection_closed(self, connection, reason):
        """Called when the connection is closed."""
        self.logger.warning(f"⚠️ Connection closed: {reason}")
        self._connection_state = 'disconnected'
        self._cleanup(force=True)
        if not self._is_shutting_down:
            self.logger.info("✨ Attempting to reconnect...")
            self.connect()
    
    def _on_channel_open(self, channel):
        """Called when the channel is opened."""
        self.logger.info("✨ Channel opened")
        self.channel = channel
        self._connection_state = 'channel_open'
        
        # Declare quote exchange
        self.channel.exchange_declare(
            exchange=self.exchange_name,
            exchange_type='fanout',
            durable=True,
            callback=self._on_quote_exchange_declared
        )
    
    def _on_quote_exchange_declared(self, frame):
        """Called when the quote exchange is declared."""
        self.logger.info(f"✨ Quote exchange {self.exchange_name} declared")
        self._connection_state = 'quote_exchange_declared'
        
        # Declare day-started exchange
        self.channel.exchange_declare(
            exchange=self.day_started_exchange,
            exchange_type='fanout',
            durable=True,
            callback=self._on_day_started_exchange_declared
        )
    
    def _on_day_started_exchange_declared(self, frame):
        """Called when the day-started exchange is declared."""
        self.logger.info(f"✨ Day-started exchange {self.day_started_exchange} declared")
        self._connection_state = 'day_started_exchange_declared'
        
        # Set up day-started consumer if callback is set
        if self._day_started_callback:
            self._setup_day_started_consumer()
        else:
            self.logger.warning("⚠️ No day-started callback set, skipping consumer setup for now (will be setup on first publish)")
            self._event.set()  # Signal that setup is complete
    
    def _on_day_started_queue_declared(self, frame):
        """Called when the day-started queue is declared."""
        self.logger.info(f"✨ Day-started queue {self.day_started_queue} declared")
        
        # Bind queue to exchange
        self.channel.queue_bind(
            exchange=self.day_started_exchange,
            queue=self.day_started_queue,
            routing_key='',
            callback=self._on_day_started_queue_bound
        )
    
    def _on_day_started_queue_bound(self, frame):
        """Called when the day-started queue is bound."""
        self.logger.info("✨ Day-started queue bound to exchange")
        
        # Start consuming
        self.channel.basic_consume(
            queue=self.day_started_queue,
            on_message_callback=self._on_day_started_message,
            auto_ack=True
        )
        
        self._event.set()
        self.logger.info("✅ Day-started event consumer is ready")
    
    def _on_day_started_message(self, channel, method, properties, body):
        """Called when a day-started message is received."""
        try:
            message = json.loads(body)
            self.logger.info(f"✨ Received day-started event: {message}")
            
            if self._day_started_callback:
                self._day_started_callback(message)
            else:
                self.logger.warning("⚠️ No callback set for day-started events")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing day-started message: {str(e)}")
    
    def connect(self):
        """Establish connection to RabbitMQ with retries."""
        if self._is_shutting_down:
            return
        
        with self._lock:
            for attempt in range(5):  # Max 5 retries
                try:
                    self.logger.info(f"✨ Attempting to connect to RabbitMQ (attempt {attempt + 1}/5)")
                    self.logger.debug(f"Connection parameters: host={self.host}, port={self.port}")
                    
                    # Test TCP connection first
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(10)  # Increased timeout
                        sock.connect((self.host, self.port))
                        sock.close()
                        self.logger.info("✨ TCP connection test successful")
                    except socket.error as e:
                        self.logger.error(f"❌ TCP connection failed: {str(e)}")
                        self.logger.error(f"Host: {self.host}, Port: {self.port}")
                        self.logger.error("Please check if RabbitMQ is running and accessible")
                        if attempt == 4:  # Last attempt
                            raise
                        time.sleep(5)  # Wait before retrying
                        continue
                    
                    self._event.clear()
                    self._connection_state = 'connecting'
                    self.connection = pika.SelectConnection(
                        parameters=self.connection_params,
                        on_open_callback=self._on_connection_open,
                        on_open_error_callback=lambda conn, err: self.logger.error(f"❌ Connection open failed: {str(err)}"),
                        on_close_callback=self._on_connection_closed
                    )
                    
                    # Start the IO loop in a separate thread
                    if self._io_thread is None or not self._io_thread.is_alive():
                        self._io_thread = threading.Thread(
                            target=self._run_io_loop,
                            name="RabbitMQ-IOLoop",
                            daemon=True
                        )
                        self._io_thread.start()
                    
                    # Wait for connection setup with increased timeout
                    if self._event.wait(timeout=30):  # Increased from 10
                        self.logger.info("✅ Successfully connected to RabbitMQ")
                        return
                    else:
                        self.logger.warning(f"⚠️ Timeout waiting for connection setup. Current state: {self._connection_state}")
                        self._cleanup(force=True)
                        
                except Exception as e:
                    if attempt == 4:  # Last attempt
                        self.logger.error(f"❌ Failed to connect to RabbitMQ after 5 attempts: {str(e)}")
                        self.logger.error("Please check the following:")
                        self.logger.error("1. Is RabbitMQ running?")
                        self.logger.error("2. Are the host and port correct?")
                        self.logger.error("3. Is the RabbitMQ server accessible from this container?")
                        self.logger.error("4. Are the credentials correct?")
                        raise
                    self.logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(5)  # Wait 5 seconds before retrying
    
    def _run_io_loop(self):
        """Run the IO loop in a separate thread."""
        try:
            self.logger.info("✨ Starting RabbitMQ IO loop")
            self.connection.ioloop.start()
        except Exception as e:
            self.logger.error(f"❌ IO loop terminated with error: {str(e)}")
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
            self.logger.warning("⚠️ Cannot publish during shutdown")
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
                            self.logger.error(f"❌ Failed to establish connection for publishing: {str(e)}")
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
                        "MessageType": "StockQuote",  # Add message type for MassTransit
                        "ModelType": prediction.get('model_type', 'unknown'),  # Add model type
                        "Confidence": float(prediction.get('confidence_score', 0.0)),  # Add confidence score
                        "ModelVersion": prediction.get('model_version', 'unknown')  # Add model version
                    }
                    
                    # Serialize message
                    message_body = json.dumps(message).encode('utf-8')
                    
                    # Check message size
                    if len(message_body) > self.connection_params.frame_max:
                        self.logger.error(f"❌ Message size ({len(message_body)} bytes) exceeds maximum frame size")
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
                            'MT-Message-Type': 'StockQuote',
                            'MT-Model-Type': prediction.get('model_type', 'unknown')  # Add model type to headers
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
                    
                    self.logger.info(
                        f"✅ Published prediction for {symbol} ({prediction.get('model_type', 'unknown')}): "
                        f"${prediction.get('prediction', 0.0):.2f} (Confidence: {prediction.get('confidence_score', 0.0):.1%})"
                    )
                    return True
                    
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        self.logger.error(f"❌ Failed to publish message for {symbol} after 3 attempts: {str(e)}")
                        return False
                    self.logger.warning(f"⚠️ Publish attempt {attempt + 1} failed: {str(e)}")
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
                self.logger.error(f"❌ Error during forced cleanup: {str(e)}")
            else:
                self.logger.warning(f"⚠️ Error during cleanup: {str(e)}")
    
    def _cleanup_on_exit(self):
        """Clean up on application exit."""
        self._is_shutting_down = True
        self._cleanup(force=True)
    
    def close(self):
        """Close the RabbitMQ connection."""
        self._is_shutting_down = True
        self._cleanup(force=True) 