#!/bin/sh
# Script to initialize the stock database tables during container startup

# Wait for the database to be ready
echo "Waiting for stock database to be ready..."
while ! nc -z postgres-stock-data 5432; do
  sleep 1
done
echo "Stock database is ready."

# Run the database initialization script
echo "Initializing stock database tables..."
python -c "
import asyncio
import sys
sys.path.append('/app')

try:
    from services.data_ingestion.db.init_db import init_stock_db
    asyncio.run(init_stock_db())
    print('✅ Database initialization completed successfully')
except Exception as e:
    print(f'❌ Database initialization failed: {e}')
    # Don't exit with error - let the service start anyway and try to create tables if needed
"

# Start the main application
echo "Starting data ingestion service..."
exec "$@"
