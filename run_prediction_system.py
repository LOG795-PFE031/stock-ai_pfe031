import subprocess
import time
import signal
import os
import sys

def main():
    """
    Run the complete prediction system:
    1. Start Celery worker
    2. Run the prediction service
    3. Allow time for tasks to be processed
    4. Terminate the worker
    """
    print("=== Stock Prediction System with Celery and RabbitMQ ===")
    
    # Start Celery worker in the background
    print("\n1. Starting Celery worker...")
    celery_cmd = [
        "celery", "-A", "celery_tasks", "worker", 
        "-l", "info", 
        "--pool=solo"  # Use solo pool to avoid subprocess issues
    ]
    
    # Start the worker in a new process
    worker_process = subprocess.Popen(
        celery_cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Give time for the worker to start up
    print("   Waiting for Celery worker to initialize...")
    time.sleep(5)
    
    # Check if worker is still running
    if worker_process.poll() is not None:
        print("ERROR: Celery worker failed to start!")
        return
    
    print("   Worker started successfully!")
    
    try:
        # Run the prediction service
        print("\n2. Running prediction service...")
        prediction_cmd = ["python", "prediction_service.py"]
        prediction_process = subprocess.run(
            prediction_cmd,
            capture_output=True,
            text=True
        )
        
        # Print the prediction service output
        print("\n=== Prediction Service Output ===")
        print(prediction_process.stdout)
        
        if prediction_process.stderr:
            print("\n=== Prediction Service Errors ===")
            print(prediction_process.stderr)
        
        # Allow time for tasks to be processed
        print("\n3. Waiting for Celery tasks to complete...")
        time.sleep(5)
        
    finally:
        # Terminate the worker process
        print("\n4. Shutting down Celery worker...")
        if worker_process.poll() is None:
            os.kill(worker_process.pid, signal.SIGTERM)
            worker_process.wait(timeout=5)
            print("   Worker terminated successfully!")
        
        # Print some worker output
        worker_output = worker_process.stdout.read()
        print("\n=== Worker Output (excerpt) ===")
        # Print only the last 20 lines (or less if there are fewer lines)
        lines = worker_output.splitlines()
        for line in lines[-20:]:
            print(line)
    
    print("\n=== Stock Prediction System Complete ===")

if __name__ == "__main__":
    main()