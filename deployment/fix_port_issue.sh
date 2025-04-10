#!/bin/bash
# Script to fix the port issue on the EC2 instance

# Set your SSH credentials
EC2_HOST="54.225.9.196"
EC2_USERNAME="ubuntu"
SSH_KEY_PATH="path/to/your/key.pem"

# Create the fix script
cat > fix_port_commands.sh << 'EOL'
#!/bin/bash
set -e

echo "=== CHECKING CURRENT CONTAINER STATUS ==="
sudo docker ps -a

# Stop and remove any existing container
echo "Stopping any existing ml-model-app container..."
sudo docker stop ml-model-app 2>/dev/null || true
sudo docker rm ml-model-app 2>/dev/null || true

# Check if AWS credentials are available
if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" && -n "$AWS_DEFAULT_REGION" ]]; then
    echo "AWS credentials found. Proceeding with deployment..."
    
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    
    if [[ -n "$AWS_ACCOUNT_ID" ]]; then
        ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
        
        # Login to ECR
        echo "Logging in to Amazon ECR..."
        aws ecr get-login-password | sudo docker login --username AWS --password-stdin ${ECR_REGISTRY}
        
        # Pull latest image
        echo "Pulling latest image..."
        sudo docker pull ${ECR_REGISTRY}/hf:latest
        
        # Run the container with the CORRECT PORT (8000)
        echo "Starting new container with port 8000..."
        sudo docker run -d --name ml-model-app \
          -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
          -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
          -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
          -p 0.0.0.0:8000:8000 \
          ${ECR_REGISTRY}/hf:latest
        
        echo "Container started. Checking status..."
        sleep 5
        sudo docker ps | grep ml-model-app || echo "Container failed to start or exited!"
        
        echo "Container logs:"
        sudo docker logs ml-model-app
        
        # Check if the application is listening on port 8000
        echo "Checking if application is listening on port 8000..."
        netstat -tulpn | grep 8000 || echo "No process is listening on port 8000!"
        
        # Try to access the application locally
        echo "Trying to access the application locally..."
        curl -v http://localhost:8000/ || echo "Failed to access application locally!"
    else
        echo "Failed to get AWS account ID. Check your AWS credentials."
    fi
else
    echo "AWS credentials not provided. Cannot deploy."
fi

echo "Fix completed!"
EOL

# Copy the script to EC2 and execute it
echo "Copying fix script to EC2..."
scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no fix_port_commands.sh ${EC2_USERNAME}@${EC2_HOST}:~/fix_port_commands.sh

echo "Making script executable..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} "chmod +x ~/fix_port_commands.sh"

echo "Running fix script on EC2..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} \
  "AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID}' \
   AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY}' \
   AWS_DEFAULT_REGION='${AWS_DEFAULT_REGION}' \
   bash ~/fix_port_commands.sh" | tee fix_port_results.txt

echo "Fix script completed! Results saved to fix_port_results.txt"
