#!/bin/bash
# Script to check what port the application is using

# Set your SSH credentials
EC2_HOST="54.225.9.196"
EC2_USERNAME="ubuntu"
SSH_KEY_PATH="path/to/your/key.pem"

# Create the port check script
cat > port_check_commands.sh << 'EOL'
#!/bin/bash

echo "=== CHECKING ALL LISTENING PORTS ==="
echo "All TCP ports in LISTEN state:"
sudo netstat -tulpn | grep LISTEN

echo ""
echo "=== CHECKING DOCKER CONTAINER PORTS ==="
echo "Docker container port mappings:"
sudo docker ps --format "{{.Names}} -> {{.Ports}}"

echo ""
echo "=== CHECKING DOCKER CONTAINER CONFIGURATIONS ==="
for container in $(sudo docker ps -q); do
  echo "Container: $(sudo docker inspect --format='{{.Name}}' $container)"
  echo "Port bindings: $(sudo docker inspect --format='{{json .HostConfig.PortBindings}}' $container)"
  echo "Exposed ports: $(sudo docker inspect --format='{{json .Config.ExposedPorts}}' $container)"
  echo ""
done

echo "=== CHECKING APPLICATION CODE FOR PORT CONFIGURATION ==="
# Check common files that might contain port configuration
for container in $(sudo docker ps -q); do
  container_name=$(sudo docker inspect --format='{{.Name}}' $container | sed 's/\///')
  echo "Checking container: $container_name"
  
  # Create a temporary script to run inside the container
  cat > /tmp/check_port.sh << 'EOF'
#!/bin/bash
echo "Searching for port configuration in common files..."

# Check for common port configuration patterns
grep -r "port" --include="*.py" --include="*.js" --include="*.json" --include="*.conf" /app 2>/dev/null || echo "No port configuration found in /app"
grep -r "PORT" --include="*.py" --include="*.js" --include="*.json" --include="*.conf" /app 2>/dev/null || echo "No PORT configuration found in /app"
grep -r "8080" --include="*.py" --include="*.js" --include="*.json" --include="*.conf" /app 2>/dev/null || echo "No 8080 references found in /app"
grep -r "8000" --include="*.py" --include="*.js" --include="*.json" --include="*.conf" /app 2>/dev/null || echo "No 8000 references found in /app"

# Check for common configuration files
for file in app.py main.py server.py index.js app.js config.json package.json; do
  if [ -f "/app/$file" ]; then
    echo "Found $file:"
    cat "/app/$file"
    echo ""
  fi
done
EOF
  
  # Copy the script to the container and run it
  sudo docker cp /tmp/check_port.sh $container_name:/tmp/check_port.sh
  sudo docker exec $container_name chmod +x /tmp/check_port.sh
  sudo docker exec $container_name /tmp/check_port.sh
  echo ""
done

echo "=== PORT CHECK COMPLETED ==="
EOL

# Copy the script to EC2 and execute it
echo "Copying port check script to EC2..."
scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no port_check_commands.sh ${EC2_USERNAME}@${EC2_HOST}:~/port_check_commands.sh

echo "Making script executable..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} "chmod +x ~/port_check_commands.sh"

echo "Running port check script on EC2..."
ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${EC2_USERNAME}@${EC2_HOST} "bash ~/port_check_commands.sh" | tee port_check_results.txt

echo "Port check completed! Results saved to port_check_results.txt"
