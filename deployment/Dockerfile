FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and configuration files
COPY saved_models/ /app/saved_models/
COPY config/ /app/config/
COPY best_xgboost_model.pkl /app/

# Copy the application code
COPY app.py .
COPY HF/ /app/HF/

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]