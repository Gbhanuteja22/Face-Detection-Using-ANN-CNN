@echo off
echo Setting up Face Recognition Project...

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete!
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the application: python main.py
echo.
pause
