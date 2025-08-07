# MySQL Migration Guide

This document describes the migration from PostgreSQL to MySQL for the data_service in the Stock AI project.

## Changes Made

### 1. Configuration Updates

**File: `core/config.py`**
- Replaced `PostgresDatabaseConfig` with `MySQLDatabaseConfig`
- Updated connection URLs to use MySQL drivers:
  - Async: `mysql+aiomysql://`
  - Sync: `mysql+pymysql://`
- Updated default port from 5432 (PostgreSQL) to 3306 (MySQL)
- Updated hostname from `postgres-stock-ai` to `mysql-stock-ai`

### 2. Database Session Configuration

**File: `db/session.py`**
- Updated to use MySQL configuration
- Changed from `config.postgres` to `config.mysql`

### 3. Database Models

**Files: `db/models/stock_price.py`, `db/models/prediction.py`**
- Replaced `Numeric` with `DECIMAL` for better MySQL compatibility
- Updated data type definitions to use MySQL-compatible types

### 4. Docker Configuration

**File: `docker-compose.yml`**
- Replaced PostgreSQL service with MySQL 8.0
- Updated environment variables for MySQL
- Changed volume mount from `postgres_data` to `mysql_data`
- Updated health check to use MySQL commands
- Updated Prefect service to use MySQL connection

### 5. Dependencies

**File: `requirements.txt`**
- Removed: `psycopg2-binary==2.9.10`
- Added: `pymysql==1.1.0`, `aiomysql==0.2.0`

### 6. Git Configuration

**Files: `.gitignore`, `.dockerignore`**
- Updated to ignore `mysql_data/` instead of `postgres_data/`

## Database Schema

The database schema remains the same with two main tables:

### Stock Prices Table
```sql
CREATE TABLE stock_prices (
    id INT PRIMARY KEY AUTO_INCREMENT,
    stock_symbol VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    date DATE NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume INT,
    dividends DECIMAL(12,4),
    stock_splits DECIMAL(12,4),
    INDEX idx_stock_symbol (stock_symbol),
    INDEX idx_date (date),
    INDEX idx_stock_symbol_date (stock_symbol, date)
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    date DATE NOT NULL,
    stock_symbol VARCHAR(10) NOT NULL,
    prediction DECIMAL(12,6) NOT NULL,
    confidence DECIMAL(5,3) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_version INT NOT NULL
);
```

## Testing

To test the MySQL configuration:

1. Start the MySQL container:
   ```bash
   docker-compose up mysql -d
   ```

2. Run the test script:
   ```bash
   python test_mysql_connection.py
   ```

## Migration Notes

- All existing PostgreSQL data will need to be migrated if you have existing data
- The application will automatically create the required tables on startup
- MySQL 8.0 is used for better performance and features
- The migration maintains backward compatibility with existing API endpoints

## Benefits of MySQL

1. **Performance**: MySQL often provides better performance for read-heavy workloads
2. **Simplicity**: Easier setup and configuration
3. **Compatibility**: Better compatibility with various hosting providers
4. **Resource Usage**: Generally lower memory and CPU usage

## Troubleshooting

If you encounter connection issues:

1. Ensure MySQL container is running: `docker-compose ps`
2. Check MySQL logs: `docker-compose logs mysql`
3. Verify network connectivity: `docker-compose exec stock-ai ping mysql-stock-ai`
4. Test direct connection: `docker-compose exec mysql mysql -u admin -padmin stocks` 
