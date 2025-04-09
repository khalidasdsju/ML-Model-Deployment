HEART FAILURE PREDICTION APPLICATION
====================================

This application provides a stylish and professional web interface for predicting heart failure risk using machine learning models.

INSTALLATION
-----------
1. Make sure you have Python 3.9+ installed
2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install the required packages:
   pip install -r requirements.txt

RUNNING THE APPLICATION
----------------------
There are two ways to run the application:

1. Using FastAPI (Original Backend):
   python run_app_1010.py

2. Using Flask (Alternative Backend):
   python run_flask.py

The application will be available at:
http://localhost:1010 or http://localhost:1020 (depending on which version you run)

FEATURES
--------
- Dashboard with statistics and visualizations
- Single patient prediction form
- Batch prediction with CSV upload
- Prediction history tracking
- Model information display

TROUBLESHOOTING
--------------
- If you encounter port conflicts, edit the port number in run_app_1010.py or run_flask.py
- If you see import errors, make sure all dependencies are installed
- For data type errors, ensure your input data matches the expected format

CONTACT
-------
For support or questions, please contact:
Email: support@heartfailureprediction.com
