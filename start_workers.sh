#!/bin/bash

# Start first worker for prediction_queue_1
echo "Starting worker for prediction_queue_1..."
celery -A celery_tasks worker -Q prediction_queue_1 -n worker1@%h --loglevel=info &
WORKER1_PID=$!

# Start second worker for prediction_queue_2
echo "Starting worker for prediction_queue_2..."
celery -A celery_tasks worker -Q prediction_queue_2 -n worker2@%h --loglevel=info &
WORKER2_PID=$!

echo "Workers started with PIDs: $WORKER1_PID and $WORKER2_PID"
echo "Press Ctrl+C to stop the workers"

# Wait for user to press Ctrl+C
trap "kill $WORKER1_PID $WORKER2_PID; exit" SIGINT
wait