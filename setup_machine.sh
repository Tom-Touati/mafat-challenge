   #!/bin/bash

   # Exit on error
   set -e

   # Function to check command status
   check_command() {
      if [ $? -ne 0 ]; then
         echo "Error: $1 failed"
         exit 1
      fi
   }

   # Update system packages
   echo "Updating system packages..."
   sudo apt-get update
   sudo apt-get install -y unzip git python3.10
   check_command "Package installation"

   # Configure git
   echo "Configuring git..."
   git config --global user.name "tomtou-bspace"
   git config --global user.email "tom.touati@brain.space"

   # Setup SSH directory
   echo "Setting up SSH configuration..."
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh

   # Create SSH key file
   cat > ~/.ssh/tom_git << 'EOL'
   -----BEGIN OPENSSH PRIVATE KEY-----
   [Your SSH key content here]
   -----END OPENSSH PRIVATE KEY-----
   EOL

   chmod 600 ~/.ssh/tom_git

   # Clone repository
   echo "Cloning repository..."
   git clone https://github.com/Tom-Touati/mafat-challenge.git
   check_command "Repository cloning"

   # Install Poetry
   echo "Installing Poetry..."
   curl -sSL https://install.python-poetry.org | python3
   check_command "Poetry installation"

   # Add Poetry to PATH
   echo 'export PATH="/home/tom.touati/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc

   echo "Setup completed successfully!"
