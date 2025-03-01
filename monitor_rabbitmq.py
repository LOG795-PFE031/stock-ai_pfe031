import time
import subprocess
import os
import sys
import json
from datetime import datetime

def get_rabbitmq_info():
    """Get information about the RabbitMQ server."""
    try:
        # Get queue information
        queues_cmd = ["rabbitmqctl", "-n", "rabbitmq@localhost", "list_queues", "name", "messages_ready", "messages_unacknowledged", "--formatter", "json"]
        queues_output = subprocess.check_output(queues_cmd, stderr=subprocess.PIPE, universal_newlines=True)
        queues_data = json.loads(queues_output)
        
        # Get exchange information
        exchanges_cmd = ["rabbitmqctl", "-n", "rabbitmq@localhost", "list_exchanges", "name", "type", "--formatter", "json"]
        exchanges_output = subprocess.check_output(exchanges_cmd, stderr=subprocess.PIPE, universal_newlines=True)
        exchanges_data = json.loads(exchanges_output)
        
        # Get bindings information
        bindings_cmd = ["rabbitmqctl", "-n", "rabbitmq@localhost", "list_bindings", "source_name", "destination_name", "routing_key", "--formatter", "json"]
        bindings_output = subprocess.check_output(bindings_cmd, stderr=subprocess.PIPE, universal_newlines=True)
        bindings_data = json.loads(bindings_output)
        
        return {
            'queues': queues_data,
            'exchanges': exchanges_data,
            'bindings': bindings_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message_counts': {}
        }
    except subprocess.CalledProcessError as e:
        print(f"Error getting RabbitMQ information: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing RabbitMQ output: {e}")
        return None

def display_message_flow(data):
    """Display a visualization of the message flow in RabbitMQ."""
    if not data:
        print("No data available")
        return
    
    print(f"\n=== RabbitMQ Message Flow ({data['timestamp']}) ===\n")
    
    # Process exchanges
    exchanges = {}
    for exchange in data['exchanges']:
        # Check if exchange data is a list or dictionary
        if isinstance(exchange, dict):
            name = exchange.get('name', '')
            ex_type = exchange.get('type', '')
        elif isinstance(exchange, list) and len(exchange) >= 2:
            name = exchange[0]
            ex_type = exchange[1]
        else:
            continue
            
        if name:  # Skip empty exchange names
            exchanges[name] = {'type': ex_type, 'bindings': []}
    
    # Process bindings
    for binding in data['bindings']:
        # Check if binding data is a list or dictionary
        if isinstance(binding, dict):
            source = binding.get('source_name', '')
            destination = binding.get('destination_name', '')
            routing_key = binding.get('routing_key', '')
        elif isinstance(binding, list) and len(binding) >= 3:
            source = binding[0]
            destination = binding[1]
            routing_key = binding[2]
        else:
            continue
            
        if source and source in exchanges:
            exchanges[source]['bindings'].append({
                'destination': destination,
                'routing_key': routing_key
            })
    
    # Process queues
    queues = {}
    for queue in data['queues']:
        # Check if queue data is a list or dictionary
        if isinstance(queue, dict):
            name = queue.get('name', '')
            ready = queue.get('messages_ready', 0)
            unacked = queue.get('messages_unacknowledged', 0)
        elif isinstance(queue, list) and len(queue) >= 3:
            name = queue[0]
            ready = queue[1]
            unacked = queue[2]
        else:
            continue
            
        if name:  # Skip empty queue names
            queues[name] = {
                'ready': ready,
                'unacked': unacked,
                'total': ready + unacked
            }
    
    # Display exchanges with their bindings
    print("EXCHANGES:")
    for name, exchange in exchanges.items():
        if name:  # Skip empty exchange names
            print(f"  {name} ({exchange['type']})")
            for binding in exchange['bindings']:
                dest = binding['destination']
                key = binding['routing_key']
                queue_info = ""
                if dest in queues:
                    queue = queues[dest]
                    queue_info = f" [msgs: {queue['total']}]"
                print(f"    --({key})--> {dest}{queue_info}")
    
    # Display queue details
    print("\nQUEUES:")
    for name, queue in queues.items():
        if queue['total'] > 0:
            status = "⚠️  HAS MESSAGES"
        else:
            status = "✅ Empty"
        print(f"  {name}: {queue['ready']} ready + {queue['unacked']} unacked = {queue['total']} total {status}")
    
    print("\n" + "=" * 50)

def monitor(interval=2, count=None):
    """Monitor RabbitMQ in real-time."""
    try:
        i = 0
        while count is None or i < count:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Get and display data
            data = get_rabbitmq_info()
            display_message_flow(data)
            
            # Wait for next check
            if count is None or i < count - 1:
                print(f"\nRefreshing in {interval} seconds (Press Ctrl+C to stop)...")
                time.sleep(interval)
            
            i += 1
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    # Parse command line arguments
    interval = 2  # Default refresh interval
    count = None  # Default to continuous monitoring
    
    if len(sys.argv) > 1:
        try:
            interval = float(sys.argv[1])
        except ValueError:
            print(f"Invalid interval: {sys.argv[1]}. Using default: 2 seconds.")
    
    if len(sys.argv) > 2:
        try:
            count = int(sys.argv[2])
        except ValueError:
            print(f"Invalid count: {sys.argv[2]}. Using default: continuous.")
    
    print(f"Starting RabbitMQ monitor (refresh: {interval}s, count: {'∞' if count is None else count})")
    monitor(interval, count)