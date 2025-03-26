import aio_pika
from src.core.config import Settings

class RabbitMQService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection = None
        self.channel = None
        
    async def connect(self):
        """Establish connection to RabbitMQ"""
        if not self.connection:
            self.connection = await aio_pika.connect_robust(
                host=self.settings.RABBITMQ_HOST,
                port=self.settings.RABBITMQ_PORT,
                login=self.settings.RABBITMQ_USER,
                password=self.settings.RABBITMQ_PASS
            )
            self.channel = await self.connection.channel()
            
    async def publish_message(self, queue_name: str, message: dict):
        """Publish message to queue"""
        await self.connect()
        await self.channel.default_exchange.publish(
            aio_pika.Message(body=str(message).encode()),
            routing_key=queue_name
        ) 