#!/bin/bash
# Script to deploy the application to EC2

# Replace these variables with your actual values
AWS_ACCESS_KEY_ID="your_access_key_id"
AWS_SECRET_ACCESS_KEY="your_secret_access_key"
AWS_DEFAULT_REGION="your_aws_region"
MONGODB_URL="your_mongodb_url"
EC2_HOST="54.225.9.196"
EC2_USERNAME="ec2-user"  # or "ubuntu" depending on your EC2 instance
SSH_KEY_PATH="path/to/your/key.pem"

# Create the deployment script
cat > deploy_commands.sh << 'EOL'
#!/bin/bash
set -e

# Configure AWS CLI
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"

# Login to ECR
echo "Logging in to Amazon ECR..."
aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Stop and remove existing container
echo "Stopping any existing container..."
docker stop ml-model-app 2>/dev/null || true
echo "Removing any existing container..."
docker rm ml-model-app 2>/dev/null || true

# Make sure the container is really gone
if docker ps -a | grep -q ml-model-app; then
  echo "Container still exists, forcing removal..."
  docker rm -f ml-model-app 2>/dev/null || true
fi

# Pull latest image
echo "Pulling latest image..."
docker pull ${ECR_REGISTRY}/hf:latest

# Run new container
echo "Starting new container..."
docker run -d --name ml-model-app \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
  -e MONGODB_URL="${MONGODB_URL}" \
  -p 8080:8080 \
  ${ECR_REGISTRY}/hf:latest

echo "Container started successfully."
echo "You can access the application at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
EOL

# Copy the script to EC2 and execute it
echo "Copying deployment script to EC2..."
scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no deploy_commands.sh ${EC2_USERNAME}@${EC2_HOST}:~/deploy_commands.sh

echo "Making script executable..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} "chmod +x ~/deploy_commands.sh"

echo "Running deployment script on EC2..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} \
  "AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID}' \
   AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY}' \
   AWS_DEFAULT_REGION='${AWS_DEFAULT_REGION}' \
   MONGODB_URL='${MONGODB_URL}' \
   bash ~/deploy_commands.sh"

echo "Deployment completed!"
