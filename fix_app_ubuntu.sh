#!/bin/bash
# Script to fix application on Ubuntu EC2 instance

# Set your AWS credentials and region
AWS_ACCESS_KEY_ID="your_access_key_id"
AWS_SECRET_ACCESS_KEY="your_secret_access_key"
AWS_DEFAULT_REGION="your_aws_region"
EC2_HOST="54.225.9.196"
EC2_USERNAME="ubuntu"
SSH_KEY_PATH="path/to/your/key.pem"

# Create the fix script to run on EC2
cat > fix_app_commands.sh << 'EOL'
#!/bin/bash
set -e

echo "=== SYSTEM CHECK ==="
echo "Running as user: $(whoami)"
echo "Current directory: $(pwd)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes to take effect."
    # For this script, we'll use sudo for docker commands
else
    echo "Docker is already installed: $(docker --version)"
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Installing AWS CLI..."
    sudo apt-get update
    sudo apt-get install -y unzip
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    echo "AWS CLI installed: $(aws --version)"
else
    echo "AWS CLI is already installed: $(aws --version)"
fi

# Configure AWS CLI if credentials are provided
if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" && -n "$AWS_DEFAULT_REGION" ]]; then
    echo "Configuring AWS CLI..."
    mkdir -p ~/.aws
    cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
    cat > ~/.aws/config << EOF
[default]
region = $AWS_DEFAULT_REGION
output = json
EOF
    echo "AWS CLI configured."
else
    echo "AWS credentials not provided. Skipping AWS CLI configuration."
fi

# Check firewall status
echo "Checking firewall status..."
if command -v ufw &> /dev/null; then
    echo "UFW status:"
    sudo ufw status
    
    # Check if port 8080 is allowed
    if ! sudo ufw status | grep 8080 > /dev/null; then
        echo "Opening port 8080 in UFW..."
        sudo ufw allow 8080/tcp
        echo "Port 8080 opened in UFW."
    else
        echo "Port 8080 is already allowed in UFW."
    fi
else
    echo "UFW not found. No firewall configuration needed."
fi

# Check if the application is running
echo "Checking for running containers..."
sudo docker ps -a

# Stop any existing containers with the same name
echo "Stopping any existing ml-model-app container..."
sudo docker stop ml-model-app 2>/dev/null || true
sudo docker rm ml-model-app 2>/dev/null || true

# Check if we can access ECR
if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" && -n "$AWS_DEFAULT_REGION" ]]; then
    echo "Testing ECR access..."
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
    
    if [[ -n "$AWS_ACCOUNT_ID" ]]; then
        echo "AWS Account ID: $AWS_ACCOUNT_ID"
        ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
        
        # Login to ECR
        echo "Logging in to Amazon ECR..."
        aws ecr get-login-password | sudo docker login --username AWS --password-stdin ${ECR_REGISTRY}
        
        # Check if the hf repository exists
        echo "Checking for 'hf' repository..."
        if aws ecr describe-repositories --repository-names hf 2>/dev/null; then
            echo "Repository 'hf' found. Pulling latest image..."
            sudo docker pull ${ECR_REGISTRY}/hf:latest
            
            # Run the container with explicit port binding
            echo "Running container with explicit port binding..."
            sudo docker run -d --name ml-model-app \
              -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
              -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
              -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
              -p 0.0.0.0:8080:8080 \
              ${ECR_REGISTRY}/hf:latest
            
            echo "Container started. Checking status..."
            sleep 5
            sudo docker ps | grep ml-model-app || echo "Container failed to start or exited!"
            
            echo "Container logs:"
            sudo docker logs ml-model-app
            
            # Check if the application is listening on port 8080
            echo "Checking if application is listening on port 8080..."
            netstat -tulpn | grep 8080 || echo "No process is listening on port 8080!"
            
            # Try to access the application locally
            echo "Trying to access the application locally..."
            curl -v http://localhost:8080/ || echo "Failed to access application locally!"
        else
            echo "Repository 'hf' not found in ECR. Please check your ECR repository."
        fi
    else
        echo "Failed to get AWS account ID. Check your AWS credentials."
    fi
else
    echo "AWS credentials not provided. Cannot access ECR."
fi

# Check if port 8080 is actually the correct port
echo "Checking for other common web ports..."
for port in 80 8000 3000 5000; do
    echo "Checking port $port..."
    if netstat -tulpn | grep ":$port " > /dev/null; then
        echo "Found process listening on port $port!"
        netstat -tulpn | grep ":$port "
    else
        echo "No process found on port $port."
    fi
done

# Check application logs in common locations
echo "Checking for application logs..."
for log_file in /var/log/syslog /var/log/docker.log /var/log/app.log; do
    if [ -f "$log_file" ]; then
        echo "Last 10 lines of $log_file:"
        sudo tail -10 "$log_file"
    fi
done

echo "Fix script completed!"
EOL

# Copy the script to EC2 and execute it
echo "Copying fix script to EC2..."
scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no fix_app_commands.sh ${EC2_USERNAME}@${EC2_HOST}:~/fix_app_commands.sh

echo "Making script executable..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} "chmod +x ~/fix_app_commands.sh"

echo "Running fix script on EC2..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} \
  "AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID}' \
   AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY}' \
   AWS_DEFAULT_REGION='${AWS_DEFAULT_REGION}' \
   bash ~/fix_app_commands.sh" | tee fix_app_results.txt

echo "Fix script completed! Results saved to fix_app_results.txt"
