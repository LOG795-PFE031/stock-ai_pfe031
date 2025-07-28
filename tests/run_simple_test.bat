@echo off
echo Installing required packages...
pip install httpx

echo.
echo Running simple API tests from tests directory...
python test_simple_api.py

pause
