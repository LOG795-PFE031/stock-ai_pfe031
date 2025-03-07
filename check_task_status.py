import sys
import json
from celery_tasks import app

def check_task_status(task_id):
    """
    Check the status of a Celery task
    """
    print(f"Checking status of task: {task_id}")
    
    # Get the AsyncResult object for this task
    result = app.AsyncResult(task_id)
    
    # Check the state
    print(f"Task state: {result.state}")
    
    # If the task succeeded, get the result
    if result.successful():
        print("Task succeeded!")
        try:
            # Format the result for better readability
            formatted_result = json.dumps(result.result, indent=2)
            print(f"Result: {formatted_result}")
        except:
            print(f"Result: {result.result}")
    elif result.failed():
        print(f"Task failed: {result.traceback}")
    else:
        print("Task is still pending or running...")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        check_task_status(task_id)
    else:
        print("Please provide a task ID as an argument.")
        print("Usage: python check_task_status.py <task_id>")
        sys.exit(1)