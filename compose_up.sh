#!/bin/bash
set -e

# Function to check and download tar files if needed
check_and_download_tars() {
    echo "Checking for required tar files..."
    
    # Create the directory if it doesn't exist
    mkdir -p ./config/conda_tars
    
    # Check for feature_splatting2.tar
    if [ ! -f "./config/conda_tars/feature_splatting2.tar" ]; then
        echo "Downloading feature_splatting2.tar..."
        curl -L -o ./config/conda_tars/feature_splatting2.tar https://s3.us-east-1.wasabisys.com/dokustorage/conda_tars/feature_splatting2.tar
        echo "feature_splatting2.tar already exists"
    fi
    
    # Check for hoist2.tar
    if [ ! -f "./config/conda_tars/hoist2.tar" ]; then
        echo "Downloading hoist2.tar..."
        curl -L -o ./config/conda_tars/hoist2.tar https://s3.us-east-1.wasabisys.com/dokustorage/conda_tars/hoist2.tar
    else
        echo "hoist2.tar already exists"
    fi

    # Check for Miniconda3-latest-Linux-x86_64.sh
    if [ ! -f "./config/conda_tars/Miniconda3-latest-Linux-x86_64.sh" ]; then
        echo "Downloading Miniconda3-latest-Linux-x86_64.sh..."
        curl -L -o ./config/Miniconda3-latest-Linux-x86_64.sh https://s3.us-east-1.wasabisys.com/dokustorage/conda_tars/Miniconda3-latest-Linux-x86_64.sh
    else
        echo "Miniconda3-latest-Linux-x86_64.sh already exists"
    fi
    
    echo "All required tar files are ready"
}

# Check if rebuild argument is provided
if [ "$1" = "rebuild" ]; then
    echo "Rebuilding images locally..."
    
    # Check and download tar files if needed
    check_and_download_tars
    
    # 1. Build the base image first
    docker compose build base

    # 2. Start the config (HTTP server) in the background
    CONFIG_CONTAINER_ID=$(docker build -t config_server ./config/ && docker run -d -p 8000:8000 -v ./config/conda_tars:/data config_server)

    # 3. Wait for the HTTP server to be ready
    echo "Waiting for config HTTP server to be ready..."
    until curl -s http://127.0.0.1:8000/ > /dev/null; do
      sleep 1
    done
    echo "Config HTTP server is up!"

    # 4. Build the images that need the HTTP server with --network=host
    docker build --network=host -t doku88/open_world_risk_main:latest -f Dockerfile .
    docker build --network=host -t doku88/open_world_risk_hoist_former:latest -f hoist_former/Dockerfile hoist_former/

    # 5. Stop and remove the config server container
    echo "Stopping config HTTP server..."
    docker stop $CONFIG_CONTAINER_ID
    docker rm $CONFIG_CONTAINER_ID

    # 6. Update docker-compose.yml to use local images (already using local names by default)
    echo "Using locally built images"

else
    echo "Using pre-built images from Docker Hub..."
    
    # 1. Pull the pre-built images from Docker Hub in parallel
    # NOTE: doku88/train:latest image is just doku88/open_world_risk_main:latest with wandb, cvxpy, etc. installed
    # No Dockerfile exists for it yet
    echo "Pulling pre-built images from Docker Hub in parallel..."
    docker pull doku88/my-shared-base:latest &
    docker pull doku88/open_world_risk_main:latest &
    docker pull doku88/open_world_risk_hoist_former:latest &
    docker pull doku88/train:latest &
    
    # Wait for all pulls to complete
    wait
    echo "All images pulled successfully!"

    # 2. Update docker-compose.yml to use Docker Hub images
    #echo "Updating docker-compose.yml to use Docker Hub images..."
    #sed -i.bak 's|image: my-shared-base|image: doku88/my-shared-base:latest|g' docker-compose.yml
    #sed -i.bak 's|image: open_world_risk_main|image: doku88/open_world_risk_main:latest|g' docker-compose.yml
    #sed -i.bak 's|image: open_world_risk_hoist_former|image: doku88/open_world_risk_hoist_former:latest|g' docker-compose.yml
fi

# 3. Bring up all services
echo "Starting services..."
docker compose up 