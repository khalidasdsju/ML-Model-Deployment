# Deployment Guide

## Docker Deployment

### Build the Docker Image

```bash
cd deployment
docker build -t hf-prediction-api .
```

### Run with Docker Compose

```bash
cd deployment
docker-compose up -d
```

## AWS Deployment

### Deploy to EC2

1. Configure AWS credentials
2. Run the deployment script:

```bash
cd deployment
./deploy_to_ec2.sh
```

### Deploy with CI/CD

The project includes GitHub Actions workflows for CI/CD deployment to AWS.

See `.github/workflows/aws.yml` for details.
