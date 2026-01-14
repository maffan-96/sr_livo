# Sonata ROS Bridge Installation Guide

## Issue: "No module named 'pointcept'"

If you get the error: `Sonata not available: No module named 'pointcept'`, this means the Pointcept library (which Sonata/PTv3 is built on) is not installed.

## Solution: Install Pointcept

### Option 1: Install Pointcept from Source (Recommended)

Pointcept is the underlying library for PTv3/Sonata. Install it inside your Docker container:

```bash
# Inside your PTv3 Docker container
cd /workspace

# Clone Pointcept repository
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept

# Install dependencies
pip install -r requirements.txt

# Install Pointcept
pip install -e .
```

### Option 2: Install with Sonata

If you have Sonata source code:

```bash
cd /path/to/sonata
pip install -e ".[pointcept]"  # Install with pointcept dependencies
```

### Option 3: Manual Package Installation

```bash
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-geometric
pip install pointcept
```

## Verify Installation

After installation, verify it works:

```python
python3 -c "from pointcept.models.point import Point; print('Pointcept installed successfully')"
```

## Check Sonata Installation

```python
python3 -c "import sonata; print(f'Sonata version: {sonata.__version__}')"
```

## Common Issues

### 1. CUDA Version Mismatch

If you get CUDA errors, make sure PyTorch is installed with the correct CUDA version:

```bash
# Check your CUDA version
nvcc --version

# Install PyTorch with matching CUDA version (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Pointcept Build Errors

If compilation fails, ensure you have build tools:

```bash
apt-get update
apt-get install -y build-essential python3-dev
```

### 3. Import Errors After Installation

If imports still fail, try:

```bash
# Uninstall and reinstall
pip uninstall pointcept sonata -y
pip install --no-cache-dir pointcept sonata
```

## Docker Setup

If you're using Docker, add this to your Dockerfile:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Pointcept
WORKDIR /workspace
RUN git clone https://github.com/Pointcept/Pointcept.git && \
    cd Pointcept && \
    pip3 install -r requirements.txt && \
    pip3 install -e .

# Install Sonata
RUN pip3 install sonata-ssl

# Install ROS packages
RUN apt-get install -y \
    ros-noetic-sensor-msgs \
    python3-numpy \
    python3-sklearn
```

## Test the Installation

Once everything is installed, test the node:

```bash
# In one terminal, start roscore
roscore

# In another terminal (inside Docker if applicable)
rosrun sonata_ros_bridge sonata_feature_node.py

# Check for successful initialization:
# You should see:
#   [INFO] Sonata Feature Extraction Node initialized
#   [INFO]   Model: facebook/sonata
#   [INFO]   Device: cuda
#
# You should NOT see:
#   [WARN] Sonata not available: No module named 'pointcept'
```

## Alternative: Use Dict Format (Temporary)

The code now includes a fallback that tries to work without the Point class. However, this may not work with all Sonata versions. If you see warnings like:

```
[WARN] Point class not available, using dict format (may not work with all Sonata versions)
```

You should still install Pointcept for full compatibility.

## Quick Install Script

Here's a one-liner to install everything (run in your Docker container):

```bash
pip install torch torchvision && \
cd /tmp && \
git clone https://github.com/Pointcept/Pointcept.git && \
cd Pointcept && \
pip install -e . && \
echo "Installation complete!"
```

## Need More Help?

- Pointcept: https://github.com/Pointcept/Pointcept
- PTv3 Paper: https://arxiv.org/abs/2312.10035
- Sonata: Check Sonata documentation for your specific version
