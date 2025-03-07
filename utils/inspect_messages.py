import pika
import json

def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        print("\n=== New Message ===")
        print(f"Content Type: {properties.content_type}")
        print(f"Headers: {properties.headers}")
        print("Message Body:")
        print(json.dumps(message, indent=2))
        print("=================\n")
    except Exception as e:
        print(f"Error processing message: {e}")
        print(f"Raw message: {body}")

# Connect to RabbitMQ
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost')
)
channel = connection.channel()

# Declare a temporary queue
result = channel.queue_declare(queue='', exclusive=True)
queue_name = result.method.queue

# Bind the queue to the exchange
channel.queue_bind(exchange='quote-exchange', queue=queue_name)

print(f"Waiting for messages from quote-exchange. To exit press CTRL+C")

# Start consuming messages
channel.basic_consume(
    queue=queue_name,
    on_message_callback=callback,
    auto_ack=True
)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()
    connection.close()
    print("\nConsumer stopped") 