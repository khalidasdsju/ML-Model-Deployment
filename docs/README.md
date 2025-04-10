# Heart Failure Prediction API

This project provides an API for predicting heart failure risk using machine learning.

## Project Structure

- `src/`: Source code
  - `api/`: API implementation
  - `models/`: Model implementation
  - `utils/`: Utility functions
  - `data/`: Data processing
  - `config/`: Configuration files
  - `tests/`: Tests
  - `notebooks/`: Jupyter notebooks
  - `scripts/`: Utility scripts
  - `static/`: Static files
  - `templates/`: HTML templates
- `deployment/`: Deployment scripts
- `docs/`: Documentation
- `data/`: Data files
- `models/`: Model files

## Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python src/api/app.py
```

## API Usage

The API provides endpoints for predicting heart failure risk:

- `/predict`: Predict for a single patient
- `/batch-predict`: Predict for multiple patients
- `/batch-predict-file`: Predict from a CSV file

## Deployment

See the `deployment/` directory for deployment scripts.
