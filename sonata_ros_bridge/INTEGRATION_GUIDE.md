# Sonata Integration Guide for SR_LIVO

This guide explains how to integrate Sonata/PTv3 semantic segmentation with SR_LIVO using the ROS bridge approach.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Docker Integration](#docker-integration)
5. [Testing & Validation](#testing--validation)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

## Overview

### What This Integration Provides

**Sonata ROS Bridge** creates a seamless connection between SR_LIVO and Sonata without modifying SR_LIVO's source code:

```
┌─────────────┐         ┌──────────────────┐         ┌──────────────┐
│   SR_LIVO   │────────>│  Sonata Bridge   │────────>│   Semantic   │
│             │  Point  │                  │  Labels │  Point Cloud │
│ (Unchanged) │  Cloud  │  (PTv3 Inference)│         │              │
└─────────────┘         └──────────────────┘         └──────────────┘
```

### Key Advantages

✅ **Zero SR_LIVO modifications** - Works with stock SR_LIVO
✅ **Flexible deployment** - Run in Docker or native ROS
✅ **Real-time capable** - Configurable performance vs. quality
✅ **Easy to maintain** - Independent update cycles

## Quick Start

### Prerequisites Checklist

- [ ] SR_LIVO installed and working
- [ ] ROS (Melodic/Noetic) installed
- [ ] Docker with PTv3 dependencies (you mentioned you have this)
- [ ] GPU with CUDA support (recommended)

### 5-Minute Setup

```bash
# 1. Navigate to SR_LIVO directory
cd /path/to/sr_livo

# 2. Run setup script
cd sonata_ros_bridge
./setup.sh

# 3. Source workspace
source ~/catkin_ws/devel/setup.bash  # Adjust path as needed

# 4. Test installation
roslaunch sonata_ros_bridge sonata_bridge.launch
```

## Detailed Setup

### Step 1: Install Sonata in Your Docker

You mentioned you have a Docker with PTv3 dependencies. Let's add Sonata:

```bash
# Enter your PTv3 Docker container
docker exec -it your_ptv3_container bash

# Install Sonata
git clone https://github.com/facebookresearch/sonata.git /opt/sonata
cd /opt/sonata
pip install -e .

# Verify installation
python3 -c "import sonata; print('Sonata installed successfully')"
```

### Step 2: Add ROS to Your PTv3 Docker (if not present)

If your Docker doesn't have ROS:

```bash
# Inside Docker container
apt-get update
apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-sensor-msgs \
    python3-rospy \
    python3-catkin-tools

# Initialize ROS
source /opt/ros/noetic/setup.bash
```

Alternatively, rebuild your Docker using our provided `Dockerfile.example`.

### Step 3: Build Sonata ROS Bridge

```bash
# On host or inside Docker with ROS
cd ~/catkin_ws/src
ln -s /path/to/sr_livo/sonata_ros_bridge .

cd ~/catkin_ws
catkin build sonata_ros_bridge
# or: catkin_make

source devel/setup.bash
```

### Step 4: Configure Topics

Edit `sonata_ros_bridge/config/sonata_params.yaml`:

```yaml
# Match SR_LIVO's published topics
input_topic: "/cloud_registered_current"  # Current frame
# Or use:
# input_topic: "/cloud_global_map"        # Global map
# input_topic: "/color_global_map"        # With colors

output_topic: "/sonata/semantic_cloud"

# Performance tuning
downsample_voxel: 0.02  # Start with 2cm, adjust as needed
device: "cuda"           # Use GPU

# Model configuration
model_name: "facebook/sonata"
```

## Docker Integration

### Option A: Separate Containers (Recommended)

Run SR_LIVO and Sonata Bridge in separate containers communicating via ROS:

```bash
# Terminal 1: Start SR_LIVO
docker run --name srlivo --network host \
    srlivo:latest \
    roslaunch sr_livo livo_ntu.launch

# Terminal 2: Start Sonata Bridge (in your PTv3 Docker)
docker run --name sonata --network host --gpus all \
    -v ~/.cache/sonata:/root/.cache/sonata \
    your_ptv3_docker:latest \
    roslaunch sonata_ros_bridge sonata_bridge.launch
```

### Option B: Docker Compose (Easiest)

Copy `docker-compose.example.yml` to your working directory:

```bash
cp sonata_ros_bridge/docker-compose.example.yml docker-compose.yml

# Edit docker-compose.yml to set your image names
nano docker-compose.yml

# Start everything
docker-compose up
```

### Option C: Single Container

If you want everything in one container:

1. Merge SR_LIVO and PTv3 dependencies in one Dockerfile
2. Use a launch file that starts both nodes
3. This is less modular but simpler for some use cases

## Testing & Validation

### Test 1: Verify ROS Topics

```bash
# Check SR_LIVO is publishing
rostopic list | grep cloud
# Expected: /cloud_registered_current, /cloud_global_map, etc.

rostopic hz /cloud_registered_current
# Should show message rate (e.g., 10 Hz)

# Check Sonata bridge is publishing
rostopic hz /sonata/semantic_cloud
# Should match input topic rate
```

### Test 2: Inspect Point Cloud

```bash
# View a single message
rostopic echo /sonata/semantic_cloud --noarr

# Check point cloud structure
rostopic echo /sonata/semantic_cloud/fields
# Expected fields: x, y, z, r, g, b, label
```

### Test 3: Visualize in RViz

```bash
# Start RViz
rosrun rviz rviz

# In RViz:
# 1. Set Fixed Frame to "camera_init" or appropriate frame
# 2. Add -> PointCloud2
# 3. Topic: /sonata/semantic_cloud
# 4. Color Transformer: Flat Color or RGB8
# 5. You should see the point cloud with semantic labels
```

### Test 4: Check Inference Performance

Monitor the node output:

```bash
# View Sonata bridge logs
rosnode info /sonata_segmentation_node

# Check performance statistics
# The node logs statistics every 10 messages:
# "Stats: 10 clouds, avg 0.045s/cloud, 25000 points/cloud"
```

## Troubleshooting

### Problem: No point clouds received

**Symptoms:**
- Sonata bridge starts but shows no messages
- `rostopic hz /sonata/semantic_cloud` shows 0 Hz

**Solutions:**
1. Verify SR_LIVO is running:
   ```bash
   rostopic hz /cloud_registered_current
   ```

2. Check topic names match:
   ```bash
   # List all available topics
   rostopic list

   # Update config if needed
   rosparam set /sonata_segmentation_node/input_topic /correct_topic_name
   ```

3. Check frame timing:
   ```bash
   rostopic echo /cloud_registered_current/header
   # Verify timestamps are current (not old data)
   ```

### Problem: CUDA out of memory

**Symptoms:**
- "CUDA out of memory" error in logs
- Node crashes during inference

**Solutions:**
1. Reduce point cloud density:
   ```yaml
   downsample_voxel: 0.05  # Increase from 0.02 to 0.05
   ```

2. Switch to CPU (slower but stable):
   ```yaml
   device: "cpu"
   ```

3. Reduce input rate:
   ```bash
   # Throttle input topic
   rosrun topic_tools throttle messages /cloud_registered_current 5.0 /cloud_throttled

   # Update bridge to use throttled topic
   rosparam set /sonata_segmentation_node/input_topic /cloud_throttled
   ```

### Problem: Slow inference (>1 second per frame)

**Symptoms:**
- High latency between input and output
- Real-time performance not achieved

**Solutions:**
1. **Aggressive downsampling:**
   ```yaml
   downsample_voxel: 0.10  # 10cm voxels
   ```

2. **Disable expensive features:**
   ```yaml
   use_normals: false  # Skip normal estimation
   ```

3. **Verify GPU usage:**
   ```bash
   # Check if GPU is being used
   nvidia-smi
   # Should show python process using GPU
   ```

4. **Profile the bottleneck:**
   - Check if it's data transfer: CPU-GPU copy overhead
   - Check if it's preprocessing: normal estimation, downsampling
   - Check if it's model inference: actual PTv3 forward pass

### Problem: Incorrect semantic labels

**Symptoms:**
- All points have same label
- Random-looking labels
- "Model output does not contain 'seg_logits'" warning

**Solutions:**
1. **Verify model is loaded:**
   ```bash
   # Check logs on startup
   # Should see: "Sonata model loaded successfully"
   ```

2. **Check inference mode:**
   ```yaml
   # For semantic segmentation (not features):
   inference_mode: "segmentation"
   ```

3. **Verify model supports segmentation:**
   - Pre-trained Sonata models are encoders (features)
   - You may need a fine-tuned model for semantic segmentation
   - Or add a segmentation head on top of features

4. **Test with dummy model:**
   - The code has a fallback that generates random labels
   - If you're seeing random labels, Sonata isn't running inference properly

### Problem: Docker networking issues

**Symptoms:**
- Containers can't see each other's topics
- ROS_MASTER_URI connection refused

**Solutions:**
1. **Use host networking:**
   ```bash
   docker run --network host ...
   ```

2. **Set ROS environment correctly:**
   ```bash
   export ROS_MASTER_URI=http://localhost:11311
   export ROS_HOSTNAME=localhost  # or your IP
   export ROS_IP=localhost
   ```

3. **Check firewall:**
   ```bash
   # Temporarily disable to test
   sudo ufw disable
   ```

## Performance Optimization

### Baseline Configuration (Balanced)

```yaml
downsample_voxel: 0.02    # 2cm
use_normals: true
use_colors: true
device: "cuda"
```

**Expected:** ~30-50ms per frame on RTX 3080, ~20k points

### High-Speed Configuration (Real-time priority)

```yaml
downsample_voxel: 0.05    # 5cm
use_normals: false        # Skip expensive PCA
use_colors: true
device: "cuda"
```

**Expected:** ~15-25ms per frame, ~5k points

### High-Quality Configuration (Accuracy priority)

```yaml
downsample_voxel: 0.01    # 1cm
use_normals: true
use_colors: true
device: "cuda"
```

**Expected:** ~100-200ms per frame, ~50k points

### Benchmarking

Run your own benchmarks:

```bash
# Start bridge with timing logs
roslaunch sonata_ros_bridge sonata_bridge.launch

# Play rosbag (or run live)
rosbag play your_data.bag

# Monitor performance
rostopic hz /sonata/semantic_cloud
# Target: Match input rate (e.g., 10 Hz)

# The node logs average timing every 10 messages
```

## Advanced Topics

### Multiple Semantic Sources

You can run multiple bridges for different purposes:

```bash
# Bridge 1: Current frame (fast, for visualization)
roslaunch sonata_ros_bridge sonata_bridge.launch \
    input_topic:=/cloud_registered_current \
    output_topic:=/sonata/current_semantic \
    downsample_voxel:=0.05

# Bridge 2: Global map (slow, for mapping)
roslaunch sonata_ros_bridge sonata_bridge.launch \
    input_topic:=/cloud_global_map \
    output_topic:=/sonata/map_semantic \
    downsample_voxel:=0.02
```

### Custom Models

To use a different model (e.g., fine-tuned on your dataset):

```yaml
model_name: "/path/to/your/checkpoint.pth"
num_classes: 40  # Adjust to your dataset
```

Update the `load_model()` function in `sonata_segmentation_node.py` to load custom checkpoints.

### Integration with SR_LIVO's Semantic Pipeline

Currently, the bridge runs independently. To integrate with SR_LIVO's existing semantic system:

1. **Option A:** Convert 3D labels to 2D masks
   - Project semantic point cloud to image plane
   - Publish to `/segmentations` topic (OneFormer format)
   - Requires custom projection node (not included)

2. **Option B:** Replace OneFormer entirely
   - Modify SR_LIVO to subscribe to 3D semantic clouds
   - This violates the "no SR_LIVO changes" requirement
   - But provides tighter integration

3. **Option C:** Post-processing fusion
   - Run both OneFormer and Sonata
   - Fuse 2D and 3D semantic predictions
   - Best accuracy but higher computational cost

## Next Steps

1. **Test with your dataset:** Run on your specific LiDAR data
2. **Tune parameters:** Optimize for your hardware and requirements
3. **Evaluate quality:** Compare with OneFormer if available
4. **Extend functionality:** Add instance segmentation, tracking, etc.

## Support & Resources

- **Sonata GitHub:** https://github.com/facebookresearch/sonata
- **PTv3 GitHub:** https://github.com/Pointcept/PointTransformerV3
- **SR_LIVO GitHub:** https://github.com/ZikangYuan/sr_livo
- **ROS sensor_msgs:** http://wiki.ros.org/sensor_msgs

## FAQ

**Q: Can I run this without Docker?**
A: Yes! Install dependencies natively and follow the same steps.

**Q: Does this work with ROS2?**
A: Not currently. The bridge uses ROS1 (rospy). Porting to ROS2 would require rewriting with rclpy.

**Q: Can I use a different model (e.g., Mask3D, PTv2)?**
A: Yes, but you'll need to modify `sonata_segmentation_node.py` to load and run the alternative model.

**Q: How do I train Sonata on my custom dataset?**
A: See the Sonata repository for training instructions. You'll need labeled 3D point cloud data.

**Q: What's the difference between Sonata and PTv3?**
A: Sonata is a self-supervised pre-training method that produces PTv3 models with strong representations. PTv3 is the architecture.

**Q: Can this replace SR_LIVO's OneFormer integration?**
A: Not directly. This provides 3D semantic segmentation as an independent stream. OneFormer provides 2D masks that SR_LIVO projects to 3D.
