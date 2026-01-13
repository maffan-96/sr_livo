#!/bin/bash
# Setup script for Sonata ROS Bridge

set -e

echo "================================================"
echo "Sonata ROS Bridge Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if ROS is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}Error: ROS is not sourced. Please run:${NC}"
    echo "  source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo -e "${GREEN}ROS Distribution: $ROS_DISTRO${NC}"

# Find catkin workspace
if [ -z "$CATKIN_WS" ]; then
    # Try to auto-detect
    if [ -d "$HOME/catkin_ws" ]; then
        CATKIN_WS="$HOME/catkin_ws"
    elif [ -d "/workspace/catkin_ws" ]; then
        CATKIN_WS="/workspace/catkin_ws"
    else
        echo -e "${YELLOW}Warning: Could not auto-detect catkin workspace${NC}"
        echo "Please set CATKIN_WS environment variable:"
        echo "  export CATKIN_WS=/path/to/your/catkin_ws"
        echo "Or specify it now:"
        read -p "Catkin workspace path: " CATKIN_WS
    fi
fi

echo -e "${GREEN}Catkin workspace: $CATKIN_WS${NC}"

# Create workspace if it doesn't exist
if [ ! -d "$CATKIN_WS/src" ]; then
    echo "Creating catkin workspace..."
    mkdir -p "$CATKIN_WS/src"
    cd "$CATKIN_WS"
    catkin init || catkin_make
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create symlink if not already in catkin workspace
if [ "$SCRIPT_DIR" != "$CATKIN_WS/src/sonata_ros_bridge" ]; then
    echo "Creating symlink in catkin workspace..."
    ln -sf "$SCRIPT_DIR" "$CATKIN_WS/src/sonata_ros_bridge"
fi

# Make Python scripts executable
echo "Making scripts executable..."
chmod +x "$SCRIPT_DIR/scripts/"*.py

# Check for required Python packages
echo ""
echo "Checking Python dependencies..."

MISSING_DEPS=()

python3 -c "import torch" 2>/dev/null || MISSING_DEPS+=("torch")
python3 -c "import sklearn" 2>/dev/null || MISSING_DEPS+=("scikit-learn")
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo -e "${YELLOW}Missing Python dependencies: ${MISSING_DEPS[@]}${NC}"
    echo "Install with:"
    echo "  pip install -r $SCRIPT_DIR/requirements.txt"
else
    echo -e "${GREEN}All Python dependencies found${NC}"
fi

# Check for Sonata
python3 -c "import sonata" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Sonata package not found${NC}"
    echo "Install Sonata from:"
    echo "  git clone https://github.com/facebookresearch/sonata.git"
    echo "  cd sonata && pip install -e ."
else
    echo -e "${GREEN}Sonata package found${NC}"
fi

# Build the workspace
echo ""
echo "Building catkin workspace..."
cd "$CATKIN_WS"

if command -v catkin &> /dev/null; then
    catkin build sonata_ros_bridge
elif command -v catkin_make &> /dev/null; then
    catkin_make
else
    echo -e "${RED}Error: Neither catkin build nor catkin_make found${NC}"
    exit 1
fi

# Source the workspace
echo ""
echo "Setup complete!"
echo ""
echo -e "${GREEN}To use the Sonata ROS Bridge:${NC}"
echo "  1. Source the workspace:"
echo "     source $CATKIN_WS/devel/setup.bash"
echo ""
echo "  2. Start SR_LIVO:"
echo "     roslaunch sr_livo livo_ntu.launch"
echo ""
echo "  3. In another terminal, start the bridge:"
echo "     roslaunch sonata_ros_bridge sonata_bridge.launch"
echo ""
echo "  4. Visualize in RViz:"
echo "     rviz"
echo "     Add PointCloud2 display, topic: /sonata/semantic_cloud"
echo ""
echo -e "${YELLOW}Note: Make sure you're running inside your PTv3 Docker environment${NC}"
