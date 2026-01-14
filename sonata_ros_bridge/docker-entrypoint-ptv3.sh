#!/bin/bash
# Entrypoint script for Sonata ROS Bridge Docker

set -e

# Source ROS Noetic
source /opt/ros/noetic/setup.bash

# Source Livox workspace if available
if [ -f /xos/ws_livox/devel/setup.bash ]; then
    source /xos/ws_livox/devel/setup.bash
fi

# Source main workspace if available
if [ -f /xos/catkin_ws/devel/setup.bash ]; then
    source /xos/catkin_ws/devel/setup.bash
fi

# Setup environment
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH

# Print environment info
echo "=========================================="
echo "Sonata ROS Bridge Docker Environment"
echo "=========================================="
echo "ROS Distribution: $ROS_DISTRO"
echo "Workspace: /xos/catkin_ws"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Check Sonata
if python3 -c "import sonata" 2>/dev/null; then
    echo "✓ Sonata installed"
else
    echo "✗ Sonata not found"
fi

# Check scikit-learn
if python3 -c "import sklearn" 2>/dev/null; then
    echo "✓ scikit-learn installed"
else
    echo "✗ scikit-learn not found"
fi

echo "=========================================="

# Execute the command
exec "$@"
