#!/bin/bash
set -e

# Source ROS setup
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source workspace setup if it exists
if [ -f /workspace/catkin_ws/devel/setup.bash ]; then
    source /workspace/catkin_ws/devel/setup.bash
fi

# Execute the command passed to docker run
exec "$@"
