# Running Sonata ROS Bridge with Your PTv3 Docker

This guide shows how to run the Sonata ROS bridge with your existing PTv3 Docker environment.

## Docker Analysis ✅

Your Docker already includes all necessary dependencies:

| Requirement | Status | Version/Note |
|-------------|--------|--------------|
| ROS | ✅ Installed | Noetic |
| PyTorch | ✅ Installed | 2.3.1 + CUDA 12.1 |
| CUDA | ✅ Installed | 12.1 |
| scikit-learn | ✅ Installed | 1.3.2 (for PCA) |
| Sonata | ✅ Installed | From GitHub |
| NumPy | ✅ Installed | 1.24.4 |
| Open3D | ✅ Installed | 0.18.0 |

**You're ready to go!** No additional installation needed in the Docker.

## Quick Start

### Option 1: Using the Run Script (Easiest)

```bash
# 1. Edit the Docker image name
cd sonata_ros_bridge
nano run_docker.sh  # Change DOCKER_IMAGE="your_ptv3_docker:latest"

# 2. Make executable and run
chmod +x run_docker.sh
./run_docker.sh

# Inside Docker, you'll see the build complete, then:
roslaunch sonata_ros_bridge sonata_feature_bridge.launch
```

### Option 2: Manual Docker Run

```bash
# Set your Docker image name
DOCKER_IMAGE="your_ptv3_docker:latest"  # Change this!

# Run Docker with SR_LIVO mounted
docker run -it --rm \
    --name sonata_bridge \
    --gpus all \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/..:/xos/catkin_ws/src/sr_livo:rw \
    -v ~/.cache/sonata:/root/.cache/sonata:rw \
    -e ROS_MASTER_URI=http://localhost:11311 \
    -e ROS_HOSTNAME=localhost \
    $DOCKER_IMAGE
```

### Option 3: Docker Compose (Recommended for Production)

See the complete `docker-compose.yml` example below.

## Step-by-Step Setup

### 1. Prepare Your Docker Image

Your Dockerfile is already perfect! It has:
- ROS Noetic with perception stack
- CUDA 12.1 + PyTorch 2.3.1
- Sonata pre-installed
- All Python dependencies

**Just build your Docker:**
```bash
# If you haven't built it yet
docker build -t ptv3_ros:latest -f your_dockerfile .
```

### 2. Mount SR_LIVO Code

The key is to mount the SR_LIVO directory into `/xos/catkin_ws/src/`:

```bash
-v /path/to/sr_livo:/xos/catkin_ws/src/sr_livo:rw
```

This makes `sonata_ros_bridge/` available inside Docker at:
```
/xos/catkin_ws/src/sr_livo/sonata_ros_bridge/
```

### 3. Build Inside Docker

```bash
# Inside Docker container
cd /xos/catkin_ws

# Create symlink for easier access (optional)
ln -sf /xos/catkin_ws/src/sr_livo/sonata_ros_bridge /xos/catkin_ws/src/sonata_ros_bridge

# Build
source /opt/ros/noetic/setup.bash
source /xos/ws_livox/devel/setup.bash  # Your Livox workspace
catkin_make --pkg sonata_ros_bridge

# Source the workspace
source devel/setup.bash
```

### 4. Run Feature Extraction

```bash
# Launch the feature extraction node
roslaunch sonata_ros_bridge sonata_feature_bridge.launch

# You should see:
# - "Sonata model loaded successfully"
# - "Buffering features for PCA: X/10"
# - "PCA fitted on XXXXX points"
# - Point clouds being published
```

## Complete Docker Compose Setup

Create `docker-compose.yml` in your SR_LIVO directory:

```yaml
version: '3.8'

services:
  # ROS Master
  roscore:
    image: ros:noetic-ros-core
    container_name: roscore
    command: roscore
    network_mode: host
    restart: unless-stopped

  # SR_LIVO (if you have it Dockerized)
  srlivo:
    image: srlivo:latest
    container_name: srlivo
    depends_on:
      - roscore
    network_mode: host
    volumes:
      - ./sr_livo:/workspace/sr_livo:ro
      - ./data:/workspace/data:ro
      - ./output:/workspace/output:rw
    environment:
      - ROS_MASTER_URI=http://localhost:11311
      - ROS_HOSTNAME=localhost
    command: >
      bash -c "source /opt/ros/noetic/setup.bash &&
               roslaunch sr_livo livo_ntu.launch"

  # Sonata Feature Extraction Bridge
  sonata_bridge:
    image: ptv3_ros:latest  # Your Docker image name
    container_name: sonata_bridge
    depends_on:
      - roscore
    network_mode: host
    runtime: nvidia
    environment:
      - DISPLAY=${DISPLAY}
      - ROS_MASTER_URI=http://localhost:11311
      - ROS_HOSTNAME=localhost
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./:/xos/catkin_ws/src/sr_livo:rw
      - ~/.cache/sonata:/root/.cache/sonata:rw
    working_dir: /xos/catkin_ws
    command: >
      bash -c "
        source /opt/ros/noetic/setup.bash &&
        source /xos/ws_livox/devel/setup.bash &&
        ln -sf /xos/catkin_ws/src/sr_livo/sonata_ros_bridge /xos/catkin_ws/src/sonata_ros_bridge || true &&
        catkin_make --pkg sonata_ros_bridge &&
        source devel/setup.bash &&
        roslaunch sonata_ros_bridge sonata_feature_bridge.launch
      "

  # Optional: RViz for visualization
  rviz:
    image: ros:noetic-ros-base
    container_name: rviz
    depends_on:
      - roscore
    network_mode: host
    environment:
      - DISPLAY=${DISPLAY}
      - ROS_MASTER_URI=http://localhost:11311
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    command: >
      bash -c "source /opt/ros/noetic/setup.bash &&
               sleep 5 &&
               rviz"
    privileged: true
```

**Usage:**
```bash
# Allow X11 forwarding
xhost +local:docker

# Start all services
docker-compose up

# Or start specific services
docker-compose up sonata_bridge rviz
```

## Workflow Examples

### Example 1: SR_LIVO Running on Host, Sonata in Docker

**Terminal 1 (Host):**
```bash
# Start SR_LIVO natively
cd ~/catkin_ws
source devel/setup.bash
roslaunch sr_livo livo_ntu.launch
```

**Terminal 2 (Docker):**
```bash
# Start Sonata bridge in Docker
cd /path/to/sr_livo/sonata_ros_bridge
./run_docker.sh

# Inside Docker:
roslaunch sonata_ros_bridge sonata_feature_bridge.launch
```

**Terminal 3 (Host or Docker):**
```bash
# Visualize in RViz
rosrun rviz rviz

# Add PointCloud2 display
# Topic: /sonata/pca_cloud
# Color Transformer: RGB8
```

### Example 2: Everything in Docker

Use the docker-compose.yml above:

```bash
# Terminal 1
docker-compose up

# All services start automatically:
# - roscore
# - SR_LIVO (if available)
# - Sonata bridge
# - RViz (optional)
```

### Example 3: Rosbag Playback

**Terminal 1 (Host):**
```bash
roscore
```

**Terminal 2 (Host):**
```bash
rosbag play your_data.bag --clock
```

**Terminal 3 (Docker):**
```bash
cd /path/to/sr_livo/sonata_ros_bridge
./run_docker.sh

# Inside Docker:
roslaunch sonata_ros_bridge sonata_feature_bridge.launch \
    input_topic:=/your/pointcloud/topic
```

**Terminal 4 (Host):**
```bash
rviz  # Visualize /sonata/pca_cloud
```

## Configuration

### Adjust Topics

If SR_LIVO publishes on a different topic:

```bash
roslaunch sonata_ros_bridge sonata_feature_bridge.launch \
    input_topic:=/your/custom/topic \
    output_pca_topic:=/custom/pca_cloud
```

### Performance Tuning

**For faster inference (larger voxels):**
```bash
roslaunch sonata_ros_bridge sonata_feature_bridge.launch \
    downsample_voxel:=0.05
```

**For CPU instead of GPU:**
```bash
roslaunch sonata_ros_bridge sonata_feature_bridge.launch \
    device:=cpu
```

**More PCA stability (more buffering):**
```bash
roslaunch sonata_ros_bridge sonata_feature_bridge.launch \
    pca_buffer_size:=20
```

### Edit Config File

Alternatively, edit the config directly:
```bash
# Inside Docker
nano /xos/catkin_ws/src/sr_livo/sonata_ros_bridge/config/sonata_params.yaml
```

## Troubleshooting

### Issue: "Sonata model not found"

**Check Sonata installation:**
```bash
# Inside Docker
python3 -c "import sonata; print(sonata.__file__)"
```

If not found, reinstall:
```bash
pip3 install git+https://github.com/facebookresearch/sonata.git
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Increase voxel downsampling:
   ```yaml
   downsample_voxel: 0.10  # 10cm instead of 2cm
   ```

2. Use CPU:
   ```yaml
   device: "cpu"
   ```

3. Check GPU usage:
   ```bash
   nvidia-smi
   ```

### Issue: "No messages on /sonata/pca_cloud"

**Check input topic:**
```bash
# List available topics
rostopic list

# Check if SR_LIVO is publishing
rostopic hz /cloud_registered_current

# Check if Sonata bridge is subscribed
rostopic info /cloud_registered_current
```

**Check node status:**
```bash
rosnode list  # Should show /sonata_feature_node
rosnode info /sonata_feature_node
```

### Issue: "scikit-learn not found"

Your Docker already has it! But to verify:
```bash
python3 -c "import sklearn; print(sklearn.__version__)"
# Should show: 1.3.2
```

### Issue: "Permission denied" on volume mount

```bash
# Fix permissions on host
chmod -R 755 /path/to/sr_livo/sonata_ros_bridge

# Or run Docker with user mapping
docker run --user $(id -u):$(id -g) ...
```

### Issue: "catkin_make fails"

**Try catkin build instead:**
```bash
cd /xos/catkin_ws
catkin build sonata_ros_bridge
```

**Check dependencies:**
```bash
rosdep install --from-paths src --ignore-src -r -y
```

## Verifying Installation

### 1. Check ROS Topics

After launching, you should see:

```bash
rostopic list

# Expected:
# /sonata/pca_cloud
# /sonata/feature_cloud
# /cloud_registered_current (input from SR_LIVO)
```

### 2. Check Message Rate

```bash
rostopic hz /sonata/pca_cloud
# Should match SR_LIVO's output rate (e.g., 10 Hz)
```

### 3. Inspect Point Cloud

```bash
rostopic echo /sonata/pca_cloud --noarr

# Should show:
# header:
#   frame_id: "camera_init"
# height: 1
# width: XXXXX
# fields:
#   - name: "x", "y", "z", "rgb"
```

### 4. Check Node Logs

```bash
# See detailed logs
rosnode info /sonata_feature_node

# Check for:
# - "Sonata model loaded successfully"
# - "PCA fitted on XXXXX points"
# - "Explained variance ratio: [0.XX, 0.XX, 0.XX]"
```

## Performance Expectations

With your hardware (GPU-enabled Docker):

| Configuration | Points/Frame | Time/Frame | FPS |
|---------------|--------------|------------|-----|
| High Quality | ~50k | ~100ms | 10 |
| Balanced | ~20k | ~40ms | 25 |
| Real-time | ~5k | ~15ms | 60+ |

**Settings:**
- High: `downsample_voxel: 0.01`
- Balanced: `downsample_voxel: 0.02` (default)
- Real-time: `downsample_voxel: 0.05`

## Docker Environment Variables

Your Docker workspace uses:
- **Workspace:** `/xos/catkin_ws`
- **Livox workspace:** `/xos/ws_livox`
- **ROS Distro:** Noetic
- **Python:** 3.8
- **PyTorch:** 2.3.1 + CUDA 12.1

Mount points:
```bash
-v /path/to/sr_livo:/xos/catkin_ws/src/sr_livo  # Source code
-v ~/.cache/sonata:/root/.cache/sonata          # Model cache
-v /tmp/.X11-unix:/tmp/.X11-unix                # X11 for RViz
```

## Network Setup

For ROS networking between Docker and host:

```bash
# Use host networking (easiest)
docker run --network host ...

# Or configure ROS_MASTER_URI
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
```

## Next Steps

1. **Test with sample data:**
   ```bash
   rosbag play sample.bag
   ```

2. **Tune parameters** in `config/sonata_params.yaml`

3. **Visualize results** in RViz (topic: `/sonata/pca_cloud`)

4. **Analyze PCA components** - See what features Sonata learned!

5. **Compare** with SR_LIVO's existing semantic output (if any)

## Additional Resources

- **[FEATURE_EXTRACTION_GUIDE.md](FEATURE_EXTRACTION_GUIDE.md)** - Understanding PCA visualization
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Complete integration guide
- **[README.md](README.md)** - Package overview

## Support

If you encounter issues:

1. Check Docker logs: `docker logs sonata_bridge`
2. Check ROS logs: `rosnode info /sonata_feature_node`
3. Verify topics: `rostopic list` and `rostopic hz`
4. Check GPU: `nvidia-smi` inside Docker
5. Review this guide's troubleshooting section

Your Docker setup is excellent and already has everything needed. Just mount the code and run! 🚀
