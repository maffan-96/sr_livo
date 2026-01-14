#!/bin/bash
# Script to run Sonata ROS Bridge with the PTv3 Docker

set -e

# Configuration
DOCKER_IMAGE="your_ptv3_docker:latest"  # Change this to your actual image name
SR_LIVO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="sonata_ros_bridge"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Sonata ROS Bridge Docker Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker image exists
if ! docker images | grep -q "$(echo $DOCKER_IMAGE | cut -d: -f1)"; then
    echo -e "${RED}Error: Docker image $DOCKER_IMAGE not found${NC}"
    echo "Please set DOCKER_IMAGE to your PTv3 Docker image name"
    exit 1
fi

echo -e "${YELLOW}Using Docker image: $DOCKER_IMAGE${NC}"
echo -e "${YELLOW}SR_LIVO path: $SR_LIVO_PATH${NC}"

# Stop existing container if running
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Run Docker container
echo -e "${GREEN}Starting Docker container...${NC}"
docker run -it --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    --network host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $SR_LIVO_PATH:/xos/catkin_ws/src/sr_livo:rw \
    -v $HOME/.cache/sonata:/root/.cache/sonata:rw \
    -e ROS_MASTER_URI=http://localhost:11311 \
    -e ROS_HOSTNAME=localhost \
    -w /xos/catkin_ws \
    $DOCKER_IMAGE \
    bash -c "
        echo '========================================';
        echo 'Building Sonata ROS Bridge...';
        echo '========================================';
        source /opt/ros/noetic/setup.bash;
        source /xos/ws_livox/devel/setup.bash || true;

        # Create symlink if not exists
        if [ ! -L /xos/catkin_ws/src/sonata_ros_bridge ]; then
            ln -sf /xos/catkin_ws/src/sr_livo/sonata_ros_bridge /xos/catkin_ws/src/sonata_ros_bridge;
        fi;

        # Build the package
        cd /xos/catkin_ws;
        catkin_make -DCMAKE_BUILD_TYPE=Release --pkg sonata_ros_bridge || catkin build sonata_ros_bridge;

        # Source the workspace
        source devel/setup.bash;

        echo '';
        echo '========================================';
        echo 'Sonata ROS Bridge built successfully!';
        echo '========================================';
        echo '';
        echo 'Available commands:';
        echo '  roslaunch sonata_ros_bridge sonata_feature_bridge.launch';
        echo '  roslaunch sonata_ros_bridge sonata_bridge.launch';
        echo '';
        echo 'Topics:';
        echo '  Input:  /cloud_registered_current (from SR_LIVO)';
        echo '  Output: /sonata/pca_cloud (PCA visualization)';
        echo '          /sonata/feature_cloud (original colors)';
        echo '';

        # Start bash shell
        exec bash;
    "
