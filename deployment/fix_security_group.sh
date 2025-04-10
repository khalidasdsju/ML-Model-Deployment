#!/bin/bash
# Script to check and fix EC2 security group settings

# Set your AWS credentials and region
AWS_ACCESS_KEY_ID="your_access_key_id"
AWS_SECRET_ACCESS_KEY="your_secret_access_key"
AWS_DEFAULT_REGION="your_aws_region"
EC2_INSTANCE_ID="i-0c16cd6ce8567b4f6" # Your EC2 instance ID

# Configure AWS CLI
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}"

# Get security group ID for the instance
echo "Getting security group ID for instance ${EC2_INSTANCE_ID}..."
SECURITY_GROUP_ID=$(aws ec2 describe-instances --instance-ids ${EC2_INSTANCE_ID} --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" --output text)

if [ -z "$SECURITY_GROUP_ID" ]; then
  echo "Failed to get security group ID. Check your AWS credentials and instance ID."
  exit 1
fi

echo "Found security group ID: ${SECURITY_GROUP_ID}"

# Check if port 8080 is already open
echo "Checking if port 8080 is already open..."
PORT_OPEN=$(aws ec2 describe-security-groups --group-ids ${SECURITY_GROUP_ID} --query "SecurityGroups[0].IpPermissions[?ToPort==\`8080\`]" --output text)

if [ -z "$PORT_OPEN" ]; then
  echo "Port 8080 is not open. Adding rule to allow inbound traffic on port 8080..."
  aws ec2 authorize-security-group-ingress \
    --group-id ${SECURITY_GROUP_ID} \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0

  echo "Rule added successfully!"
else
  echo "Port 8080 is already open in the security group."
fi

# Verify the security group rules
echo "Current security group rules:"
aws ec2 describe-security-groups --group-ids ${SECURITY_GROUP_ID} --query "SecurityGroups[0].IpPermissions" --output json

echo "Security group check completed!"
