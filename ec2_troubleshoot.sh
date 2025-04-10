#!/bin/bash
# Comprehensive troubleshooting script for EC2 deployment

# Set your AWS credentials and region
AWS_ACCESS_KEY_ID="your_access_key_id"
AWS_SECRET_ACCESS_KEY="your_secret_access_key"
AWS_DEFAULT_REGION="your_aws_region"
EC2_HOST="54.225.9.196"
EC2_USERNAME="ec2-user"  # or "ubuntu" depending on your EC2 instance
SSH_KEY_PATH="path/to/your/key.pem"

# Create the troubleshooting script to run on EC2
cat > troubleshoot_commands.sh << 'EOL'
#!/bin/bash
set -e

echo "=== SYSTEM INFORMATION ==="
echo "Hostname: $(hostname)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "Kernel: $(uname -r)"
echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^[ \t]*//')"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " available"}')"
echo ""

echo "=== NETWORK INFORMATION ==="
echo "Public IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "Private IP: $(hostname -I | awk '{print $1}')"
echo ""

echo "=== DOCKER STATUS ==="
if command -v docker &> /dev/null; then
    echo "Docker is installed: $(docker --version)"
    echo "Docker service status:"
    systemctl status docker | grep Active
    echo ""
    
    echo "=== DOCKER CONTAINERS ==="
    echo "Running containers:"
    docker ps
    echo ""
    echo "All containers (including stopped):"
    docker ps -a
    echo ""
    
    if docker ps -a | grep -q ml-model-app; then
        echo "=== ML MODEL APP CONTAINER DETAILS ==="
        echo "Container status:"
        docker inspect --format='{{.State.Status}}' ml-model-app
        echo "Container created: $(docker inspect --format='{{.Created}}' ml-model-app)"
        echo "Container ports: $(docker inspect --format='{{.NetworkSettings.Ports}}' ml-model-app)"
        echo ""
        
        echo "=== CONTAINER LOGS ==="
        echo "Last 50 lines of container logs:"
        docker logs --tail 50 ml-model-app
        echo ""
    else
        echo "ml-model-app container not found"
        echo ""
    fi
else
    echo "Docker is not installed"
    echo ""
fi

echo "=== AWS ECR ACCESS ==="
if command -v aws &> /dev/null; then
    echo "AWS CLI is installed: $(aws --version)"
    
    if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" && -n "$AWS_DEFAULT_REGION" ]]; then
        echo "Testing AWS ECR access..."
        AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || echo "Failed to get AWS account ID"
        if [[ -n "$AWS_ACCOUNT_ID" ]]; then
            echo "AWS Account ID: $AWS_ACCOUNT_ID"
            echo "Checking ECR repositories..."
            aws ecr describe-repositories --query "repositories[].repositoryName" --output text || echo "Failed to list ECR repositories"
        fi
    else
        echo "AWS credentials not set"
    fi
else
    echo "AWS CLI is not installed"
fi
echo ""

echo "=== PORT USAGE ==="
echo "Checking if port 8080 is in use:"
netstat -tulpn 2>/dev/null | grep 8080 || echo "Port 8080 is not in use"
echo ""

echo "=== FIREWALL STATUS ==="
if command -v firewall-cmd &> /dev/null; then
    echo "Firewalld status:"
    firewall-cmd --state
    echo "Checking if port 8080 is allowed:"
    firewall-cmd --list-ports | grep 8080 || echo "Port 8080 is not explicitly allowed in firewalld"
elif command -v ufw &> /dev/null; then
    echo "UFW status:"
    ufw status
else
    echo "No recognized firewall found (checked firewalld and ufw)"
fi
echo ""

echo "=== SECURITY GROUP INFO ==="
echo "To check security group settings, please verify in AWS Console that inbound traffic is allowed on port 8080"
echo ""

echo "=== DEPLOYMENT ATTEMPT ==="
echo "Attempting to deploy the application..."

# Configure AWS CLI if credentials are available
if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" && -n "$AWS_DEFAULT_REGION" ]]; then
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    if [[ -n "$AWS_ACCOUNT_ID" ]]; then
        ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
        
        # Login to ECR
        echo "Logging in to Amazon ECR..."
        aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_REGISTRY}
        
        # Stop and remove existing container
        echo "Stopping any existing container..."
        docker stop ml-model-app 2>/dev/null || true
        echo "Removing any existing container..."
        docker rm ml-model-app 2>/dev/null || true
        
        # Pull latest image
        echo "Pulling latest image..."
        docker pull ${ECR_REGISTRY}/hf:latest
        
        # Run new container
        echo "Starting new container..."
        docker run -d --name ml-model-app \
          -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
          -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
          -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
          -p 8080:8080 \
          ${ECR_REGISTRY}/hf:latest
        
        echo "Container started. Checking status..."
        sleep 5
        docker ps | grep ml-model-app
        
        echo "Container logs:"
        docker logs ml-model-app
    else
        echo "Failed to get AWS account ID. Check your AWS credentials."
    fi
else
    echo "AWS credentials not set. Skipping deployment."
fi
EOL

# Copy the script to EC2 and execute it
echo "Copying troubleshooting script to EC2..."
scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no troubleshoot_commands.sh ${EC2_USERNAME}@${EC2_HOST}:~/troubleshoot_commands.sh

echo "Making script executable..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} "chmod +x ~/troubleshoot_commands.sh"

echo "Running troubleshooting script on EC2..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} \
  "AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID}' \
   AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY}' \
   AWS_DEFAULT_REGION='${AWS_DEFAULT_REGION}' \
   bash ~/troubleshoot_commands.sh" | tee ec2_troubleshoot_results.txt

echo "Troubleshooting completed! Results saved to ec2_troubleshoot_results.txt"
