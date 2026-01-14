# Sonata ROS Bridge for SR_LIVO

A minimal ROS bridge to integrate **Sonata** (PTv3-based feature extraction) with **SR_LIVO** for **self-supervised 3D point cloud understanding**.

## Overview

This package provides a standalone ROS node that:
- Subscribes to point clouds published by SR_LIVO (`/cloud_registered_current`)
- Runs Sonata/PTv3 feature extraction on 3D point clouds
- **Visualizes learned features using PCA** (Principal Component Analysis)
- Publishes point clouds colored by PCA components to reveal semantic structure
- **Note**: SR_LIVO was minimally modified (30 lines) to publish body frame clouds - see [SR_LIVO_MODIFICATION.md](../SR_LIVO_MODIFICATION.md)

## Architecture

```
SR_LIVO → Point Cloud → Sonata Bridge → Feature Extraction → PCA → Colored Point Cloud
                            ↓                                      ↓
                    PTv3 Encoder (Docker)                  RGB from PC1,PC2,PC3
```

## Features

- **Feature Extraction**: Extract rich representations using pre-trained Sonata models
- **PCA Visualization**: Visualize high-dimensional features as RGB colors
- **Self-Supervised**: No labeled data required, works on any point cloud
- **Minimal Integration**: Standalone node with small SR_LIVO modification (30 lines)
- **Real-time Processing**: Uses current LiDAR sweep in body frame
- **Efficient Processing**: Voxel downsampling for real-time performance
- **Docker Ready**: Works with your existing PTv3 Docker environment
- **Configurable**: YAML-based configuration for all parameters

## Modes of Operation

### 1. Feature Extraction + PCA Visualization (Primary Mode) ✨

Extracts features and visualizes them using PCA:
- Input: 3D point cloud (xyz, rgb, normals)
- Output: Point cloud colored by first 3 PCA components
- Use case: Understanding learned semantic structure, scene analysis

```bash
roslaunch sonata_ros_bridge sonata_feature_bridge.launch
```

**Output topics:**
- `/sonata/pca_cloud` - Point cloud colored by PCA (main visualization)
- `/sonata/feature_cloud` - Original point cloud (reference)

### 2. Semantic Segmentation (Legacy Mode)

For semantic segmentation instead of features:
```bash
roslaunch sonata_ros_bridge sonata_bridge.launch
```

**Note:** Requires a fine-tuned segmentation model (not just pre-trained Sonata)

## Prerequisites

### 1. SR_LIVO Running
Ensure SR_LIVO is running and publishing point clouds on topics like:
- `/cloud_registered_current` (current frame in body frame)
- `/cloud_global_map` (global map)
- `/color_global_map` (color global map)

### 2. Sonata/PTv3 Environment
You mentioned you already have a Docker with PTv3 dependencies. Ensure it includes:

```bash
# Inside your Docker container
pip install torch torchvision
pip install sonata-ssl  # Or install from source
pip install scikit-learn  # For normal estimation
```

Alternatively, install Sonata from source:
```bash
git clone https://github.com/facebookresearch/sonata.git
cd sonata
pip install -e .
```

### 3. ROS Dependencies
```bash
# Standard ROS packages (should already be installed)
sudo apt-get install ros-$ROS_DISTRO-sensor-msgs
sudo apt-get install python3-numpy
```

## Installation

### 1. Build the Package

```bash
cd ~/catkin_ws/src/sr_livo/sonata_ros_bridge
cd ~/catkin_ws
catkin build sonata_ros_bridge
# or: catkin_make

source devel/setup.bash
```

### 2. Make Scripts Executable

```bash
chmod +x ~/catkin_ws/src/sr_livo/sonata_ros_bridge/scripts/*.py
```

## Usage

### Feature Extraction Mode (Recommended)

1. **Start SR_LIVO** (in one terminal):
```bash
roslaunch sr_livo livo_ntu.launch
```

2. **Start Sonata Feature Extraction** (in another terminal, inside your PTv3 Docker):
```bash
roslaunch sonata_ros_bridge sonata_feature_bridge.launch
```

3. **Visualize in RViz**:
```bash
rosrun rviz rviz

# Add PointCloud2 display
# Topic: /sonata/pca_cloud
# Color Transformer: RGB8
```

You should see the point cloud colored by PCA components, revealing the learned semantic structure!

### With Custom Parameters

```bash
roslaunch sonata_ros_bridge sonata_feature_bridge.launch \
    input_topic:=/cloud_registered_current \
    output_pca_topic:=/sonata/pca_cloud \
    device:=cuda \
    downsample_voxel:=0.02 \
    pca_components:=3 \
    pca_buffer_size:=10
```

### Understanding PCA Visualization

The colors in `/sonata/pca_cloud` represent learned features:
- **Similar colors** = similar learned representations
- **Distinct colors** = different semantic/geometric properties
- **RGB channels** = First 3 PCA components of high-dimensional features

📖 **See [FEATURE_EXTRACTION_GUIDE.md](FEATURE_EXTRACTION_GUIDE.md) for detailed explanation**

## Configuration

Edit `config/sonata_params.yaml` to customize:

```yaml
# Input/Output Topics
input_topic: "/cloud_registered_current"
output_topic: "/sonata/feature_cloud"        # Original colors
output_pca_topic: "/sonata/pca_cloud"        # PCA visualization

# Model Settings
model_name: "facebook/sonata"
device: "cuda"  # or "cpu"

# Processing Options
downsample_voxel: 0.02  # 2cm voxel downsampling
use_normals: true       # Estimate surface normals
use_colors: true        # Use RGB colors

# PCA Visualization
pca_components: 3       # Number of components (3 for RGB)
normalize_features: true  # Normalize before PCA
pca_buffer_size: 10     # Frames to buffer for PCA fitting
```

## Docker Integration

### Option 1: Run Bridge Inside Docker

If your PTv3 Docker has ROS installed:

```bash
# Start Docker with ROS network access
docker run --gpus all --network host \
    -v /path/to/sr_livo:/workspace/sr_livo \
    your_ptv3_docker:latest

# Inside Docker
source /opt/ros/$ROS_DISTRO/setup.bash
source /workspace/catkin_ws/devel/setup.bash
roslaunch sonata_ros_bridge sonata_bridge.launch
```

### Option 2: Docker with X11 Forwarding

```bash
docker run --gpus all --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/sr_livo:/workspace/sr_livo \
    your_ptv3_docker:latest
```

### Option 3: Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3'
services:
  srlivo:
    image: srlivo:latest
    network_mode: host
    volumes:
      - ./sr_livo:/workspace/sr_livo
    command: roslaunch sr_livo livo_ntu.launch

  sonata_bridge:
    image: ptv3:latest
    network_mode: host
    runtime: nvidia
    volumes:
      - ./sr_livo:/workspace/sr_livo
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command: roslaunch sonata_ros_bridge sonata_bridge.launch
```

Run with:
```bash
docker-compose up
```

## Topics

### Subscribed Topics
- `input_topic` (sensor_msgs/PointCloud2): Input point cloud from SR_LIVO
  - Default: `/cloud_registered_current`

### Published Topics (Feature Extraction Mode)
- `/sonata/pca_cloud` (sensor_msgs/PointCloud2): **Main output** - Point cloud colored by PCA components
  - Fields: `x, y, z, rgb`
  - RGB colors represent first 3 PCA components of learned features
  - Use this for visualization in RViz

- `/sonata/feature_cloud` (sensor_msgs/PointCloud2): Original point cloud with RGB
  - Fields: `x, y, z, rgb`
  - Original colors from SR_LIVO (reference)

### Published Topics (Segmentation Mode - Legacy)
- `/sonata/semantic_cloud` (sensor_msgs/PointCloud2): Point cloud with semantic labels
  - Fields: `x, y, z, r, g, b, label`
  - Requires fine-tuned segmentation model

## Performance Tuning

### For Real-Time Performance

1. **Increase voxel size** (faster but less detail):
```yaml
downsample_voxel: 0.05  # 5cm voxels
```

2. **Disable normal estimation** (if model doesn't need it):
```yaml
use_normals: false
```

3. **Use CPU if GPU is busy** with SR_LIVO:
```yaml
device: "cpu"
```

### For Best Quality

1. **Reduce voxel size**:
```yaml
downsample_voxel: 0.01  # 1cm voxels
```

2. **Enable all features**:
```yaml
use_normals: true
use_colors: true
```

## Troubleshooting

### Issue: "Sonata not available"
**Solution**: Install Sonata in your Python environment:
```bash
pip install sonata-ssl
# or from source:
cd /path/to/sonata && pip install -e .
```

### Issue: "CUDA out of memory"
**Solution**: Reduce point cloud size or use CPU:
```yaml
downsample_voxel: 0.05  # Increase voxel size
device: "cpu"           # Use CPU instead
```

### Issue: "No messages received"
**Solution**: Check SR_LIVO is publishing:
```bash
rostopic echo /cloud_registered_current --noarr
rostopic hz /cloud_registered_current
```

### Issue: "Model inference too slow"
**Solution**:
- Increase `downsample_voxel` to reduce points
- Use a faster model variant if available
- Ensure GPU is being used: check `device: "cuda"`

## Advanced Usage

### Multiple Point Cloud Sources

Run multiple bridge instances for different SR_LIVO topics:

```bash
# Bridge for current frame
roslaunch sonata_ros_bridge sonata_bridge.launch \
    input_topic:=/cloud_registered_current \
    output_topic:=/sonata/current_frame

# Bridge for global map
roslaunch sonata_ros_bridge sonata_bridge.launch \
    input_topic:=/cloud_global_map \
    output_topic:=/sonata/global_map
```

### Custom Semantic Classes

For datasets other than ScanNet, modify:
```yaml
num_classes: 40  # Adjust based on your model
```

### Integration with SR_LIVO's Semantic System

While this bridge operates independently, you could potentially:
1. Subscribe to `/sonata/semantic_cloud` in a custom node
2. Convert 3D semantic labels to 2D masks
3. Publish to `/segmentations` topic for SR_LIVO integration

This would require a separate conversion node (not included here).

## Citation

If you use Sonata in your research, please cite:

```bibtex
@inproceedings{wu2025sonata,
  title={Sonata: Self-Supervised Learning of Reliable Point Representations},
  author={Wu, Xiaoyang and others},
  booktitle={CVPR},
  year={2025}
}
```

For PTv3:
```bibtex
@inproceedings{wu2024ptv3,
  title={Point Transformer V3: Simpler, Faster, Stronger},
  author={Wu, Xiaoyang and others},
  booktitle={CVPR},
  year={2024}
}
```

For SR_LIVO:
```bibtex
@article{yuan2024sr_livo,
  title={SR-LIVO: LiDAR-Inertial-Visual Odometry and Mapping with Sweep Reconstruction},
  author={Yuan, Zikang and others},
  year={2024}
}
```

## License

This bridge code is provided as-is for research and development purposes.

## Support

For issues specific to:
- **Sonata/PTv3**: See [Sonata GitHub](https://github.com/facebookresearch/sonata)
- **SR_LIVO**: See [SR_LIVO repository](https://github.com/ZikangYuan/sr_livo)
- **This bridge**: Open an issue in the SR_LIVO repository

## Roadmap

Future enhancements:
- [ ] Support for instance segmentation
- [ ] Integration with SR_LIVO's `/segmentations` topic
- [ ] Pre-built Docker image with all dependencies
- [ ] Support for multiple model backends (PTv3, Mask3D, etc.)
- [ ] Real-time performance benchmarking tools
