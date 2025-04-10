# Heart Failure Prediction API

![Heart Failure Prediction](https://img.shields.io/badge/ML-Heart%20Failure%20Prediction-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)
![AWS](https://img.shields.io/badge/Cloud-AWS-orange)

## Overview

This project provides an API for predicting heart failure risk using machine learning. It uses a trained XGBoost model to predict the probability of heart failure based on patient data.

## Features

- REST API for heart failure prediction
- Single and batch prediction endpoints
- CSV file upload for batch predictions
- Model versioning and loading from S3
- Docker deployment
- AWS EC2 deployment
- CI/CD with GitHub Actions

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

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-failure-prediction.git
cd heart-failure-prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Run the API locally
python src/api/app.py
```

The API will be available at http://localhost:8000

### Docker Deployment

```bash
# Build and run with Docker
cd deployment
docker-compose up -d
```

## Documentation

- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Project Documentation](docs/README.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
