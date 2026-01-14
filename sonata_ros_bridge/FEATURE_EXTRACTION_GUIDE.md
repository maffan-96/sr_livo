# Sonata Feature Extraction and PCA Visualization Guide

This guide explains how to use Sonata for **feature extraction** and **PCA-based visualization** instead of semantic segmentation.

## What is Feature Extraction Mode?

Sonata is primarily a **self-supervised representation learning** method that produces rich feature embeddings for 3D point clouds. Instead of predicting semantic labels, this mode:

1. **Extracts high-dimensional features** (typically 256-512 dimensions) from the Sonata encoder
2. **Applies PCA** (Principal Component Analysis) to reduce features to 3 dimensions
3. **Visualizes features** by mapping the first 3 PCA components to RGB colors

This reveals the **learned semantic structure** in the point cloud based on Sonata's self-supervised representations.

## Architecture

```
Point Cloud → Sonata Encoder → Features [N × D]
                                    ↓
                               PCA [D → 3]
                                    ↓
                            RGB Visualization
```

## Quick Start

### 1. Launch Feature Extraction

```bash
# Start SR_LIVO (in one terminal)
roslaunch sr_livo livo_ntu.launch

# Start Sonata feature extraction (in another terminal, inside PTv3 Docker)
roslaunch sonata_ros_bridge sonata_feature_bridge.launch
```

### 2. Visualize in RViz

```bash
# Start RViz
rosrun rviz rviz

# Add two PointCloud2 displays:
# Display 1: /sonata/pca_cloud (colored by PCA components)
# Display 2: /sonata/feature_cloud (original colors, for reference)
```

## Published Topics

### `/sonata/pca_cloud`
- **Type:** `sensor_msgs/PointCloud2`
- **Description:** Point cloud colored by first 3 PCA components
- **Colors:** RGB = normalized PCA components [PC1, PC2, PC3]
- **Use:** Main visualization showing learned semantic structure

### `/sonata/feature_cloud`
- **Type:** `sensor_msgs/PointCloud2`
- **Description:** Original point cloud with RGB colors
- **Use:** Reference for comparison with PCA visualization

## Configuration

Edit `config/sonata_params.yaml`:

```yaml
# PCA Visualization Settings
pca_components: 3              # Number of PCA components (3 for RGB)
normalize_features: true       # Normalize features before PCA
pca_buffer_size: 10           # Frames to buffer before fitting PCA

# Processing Settings
downsample_voxel: 0.02        # 2cm voxel downsampling
use_normals: true             # Use surface normals as input
use_colors: true              # Use RGB colors as input

# Model Settings
model_name: "facebook/sonata" # Pre-trained Sonata model
device: "cuda"                # GPU device
```

### Key Parameters

#### `pca_components`
- **Default:** 3
- **Description:** Number of PCA dimensions
- **Usage:** Use 3 for RGB visualization, higher for more detailed analysis
- **Range:** 1-10 (but only first 3 are visualized as RGB)

#### `normalize_features`
- **Default:** true
- **Description:** Standardize features before PCA (zero mean, unit variance)
- **Recommendation:** Keep true for better PCA separation

#### `pca_buffer_size`
- **Default:** 10
- **Description:** Number of frames to accumulate before fitting PCA
- **Trade-off:**
  - Larger = more stable PCA, slower startup
  - Smaller = faster startup, potentially noisy PCA
- **Recommendation:** 5-20 frames depending on scene diversity

#### `downsample_voxel`
- **Default:** 0.02 (2cm)
- **Description:** Voxel size for downsampling
- **Trade-off:**
  - Smaller = more detail, slower inference
  - Larger = faster inference, less detail
- **Recommendation:** 0.02-0.05m for typical indoor/outdoor scenes

## Understanding PCA Visualization

### What Do the Colors Mean?

The RGB colors in `/sonata/pca_cloud` represent:
- **Red channel:** First principal component (PC1) - captures most variance
- **Green channel:** Second principal component (PC2)
- **Blue channel:** Third principal component (PC3)

Similar colors indicate similar learned features, suggesting:
- **Semantic similarity** (e.g., all walls have similar colors)
- **Geometric similarity** (e.g., all planar surfaces cluster together)
- **Material similarity** (e.g., vegetation vs. man-made structures)

### Expected Behavior

After the node processes ~10 frames (default buffer size):
1. **PCA is fitted** on accumulated features
2. **Explained variance** is logged (e.g., [0.45, 0.23, 0.12])
   - Higher values = more information captured by those components
   - First component typically captures 30-50% of variance
3. **Colors appear** on the point cloud showing feature clustering

### Good PCA Visualization Indicators

✅ **Clear color separation** between different object types
✅ **Consistent colors** for similar surfaces (e.g., all ground is similar color)
✅ **Smooth color transitions** within objects
✅ **High explained variance** (sum of first 3 PCs > 60%)

### Poor PCA Visualization Indicators

❌ **Random/noisy colors** → Increase `pca_buffer_size` or `downsample_voxel`
❌ **All one color** → Features not diverse, check model loading
❌ **Low explained variance** (< 40% for 3 components) → Scene too simple or features not learning well

## Examples

### Example 1: Indoor Scene (Office/Lab)

Expected PCA visualization:
- **Walls:** Similar greenish tint (planar, vertical)
- **Floor:** Brownish/reddish (planar, horizontal)
- **Tables/Desks:** Distinct from walls (different height, material)
- **Clutter:** Mixed colors (complex geometry)

Configuration:
```yaml
downsample_voxel: 0.02
pca_buffer_size: 10
normalize_features: true
```

### Example 2: Outdoor Scene (Street/Campus)

Expected PCA visualization:
- **Buildings:** Consistent colors per building
- **Trees/Vegetation:** Green tones (organic, irregular geometry)
- **Ground:** Road vs. sidewalk distinction
- **Vehicles:** Distinct from surroundings

Configuration:
```yaml
downsample_voxel: 0.05  # Larger scenes, more aggressive downsampling
pca_buffer_size: 15     # More diversity, larger buffer
```

### Example 3: Large-Scale Mapping

For large map reconstruction:
```yaml
input_topic: "/cloud_global_map"  # Use global map instead of current frame
downsample_voxel: 0.10             # More downsampling for large maps
pca_buffer_size: 20                # Larger buffer for map-scale diversity
```

## Workflow

### Initial Setup (First 10 Frames)

1. Node starts buffering features
2. Log shows: "Buffering features for PCA: 3/10 frames"
3. Point clouds published with original colors
4. After 10 frames: "PCA fitted on 250000 points"
5. Explained variance printed: [0.42, 0.25, 0.15]

### Continuous Operation

- Every new frame is transformed using fitted PCA
- PCA slowly adapts with new features (partial buffer retention)
- Point clouds colored by PCA continuously published

### Changing Scenes

If you move to a very different environment:
- Restart the node to reset PCA
- Or increase `pca_buffer_size` to adapt more slowly

## Advanced Usage

### Extracting Raw Features

If you want raw features (not PCA) for downstream processing:

Modify the node to save features or publish them separately:
```python
# In sonata_feature_node.py, add publisher
self.pub_raw_features = rospy.Publisher('/sonata/raw_features', ..., queue_size=1)
```

You can then process these features offline for:
- Clustering (K-means, DBSCAN)
- Classification
- Similarity search
- Transfer learning

### Multi-Scale Feature Extraction

To extract features at multiple layers:

```yaml
feature_layer: "final"  # or: 0, 1, 2, 3 for intermediate layers
```

Early layers capture low-level geometry, later layers capture high-level semantics.

### Feature Dimensionality

Sonata features are typically:
- **PTv3 backbone:** 256-512 dimensions
- **Different layers:** Different dimensions
- **Check logs:** "Extracted features shape: [N, D]"

### PCA Analysis

For research/analysis, you can:
1. Extract PCA components after fitting
2. Analyze explained variance across components
3. Project features to different dimensions
4. Visualize more than 3 components using other color schemes

Example analysis:
```python
# After PCA is fitted
print("Explained variance ratio:", node.pca.explained_variance_ratio_)
print("Total variance explained by 3 PCs:", node.pca.explained_variance_ratio_[:3].sum())

# Get PCA components (feature importance)
print("Component shapes:", node.pca.components_.shape)  # [3, D]
```

## Comparison with Semantic Segmentation

| Aspect | Feature + PCA | Semantic Segmentation |
|--------|---------------|----------------------|
| **Output** | Continuous colors (PCA) | Discrete labels (0-19) |
| **Training** | Pre-trained (self-supervised) | Requires labeled data |
| **Generalization** | Works on any scene | Limited to trained classes |
| **Visualization** | Smooth color transitions | Hard boundaries |
| **Usage** | Exploration, analysis | Task-specific (navigation, etc.) |
| **Speed** | Faster (no classification head) | Slower (additional layers) |

## Troubleshooting

### Issue: All points same color

**Cause:** PCA not fitted yet, or features not diverse

**Solutions:**
- Wait for buffer to fill (10 frames)
- Check log for "PCA fitted" message
- Increase `pca_buffer_size`
- Verify model is loaded: check for "Sonata model loaded successfully"

### Issue: Noisy colors

**Cause:** Feature extraction unstable or PCA on too few points

**Solutions:**
```yaml
downsample_voxel: 0.03      # Less downsampling
pca_buffer_size: 20         # More frames for stable PCA
normalize_features: true    # Ensure normalization
```

### Issue: Colors don't separate objects

**Cause:** Features not capturing semantic information

**Solutions:**
- Verify correct model is loaded (facebook/sonata)
- Check input quality: normals and colors should be provided
- Scene might be too uniform (e.g., empty room)

### Issue: Low explained variance

**Log shows:** `Explained variance: [0.15, 0.10, 0.08]`

**Cause:** Scene lacks diversity or features are high-dimensional and evenly distributed

**Solutions:**
- Increase `pca_buffer_size` to see more scene variety
- Acceptable if scene is genuinely uniform (e.g., single wall)
- Try different input topic (global map vs current frame)

### Issue: Inference too slow

**Symptoms:** Frame rate drops, >1s per frame

**Solutions:**
```yaml
downsample_voxel: 0.05  # More aggressive downsampling
use_normals: false      # Skip normal estimation
device: "cuda"          # Ensure GPU is used
```

Check with:
```bash
nvidia-smi  # Should show python process using GPU
```

## Performance Benchmarks

| Configuration | Points/Frame | Time/Frame | FPS |
|---------------|--------------|------------|-----|
| High Quality | ~50k | ~100ms | 10 |
| Balanced | ~20k | ~40ms | 25 |
| Real-time | ~5k | ~15ms | 60+ |

Hardware: RTX 3080, Intel i7-11700

## Research Applications

### 1. Scene Understanding
- Analyze learned feature space
- Discover emergent semantic categories
- Study feature evolution across scenes

### 2. Transfer Learning
- Use Sonata features for downstream tasks
- Fine-tune on specific applications
- Compare with other backbones (PointNet++, MinkowskiNet)

### 3. Visualization and Debugging
- Verify odometry quality (features should be consistent)
- Detect reconstruction artifacts (sudden color changes)
- Identify perceptually similar regions

### 4. Unsupervised Clustering
- Use PCA features for clustering
- Discover object instances without labels
- Group similar geometric/semantic structures

## Citation

If you use Sonata feature extraction in your research:

```bibtex
@inproceedings{wu2025sonata,
  title={Sonata: Self-Supervised Learning of Reliable Point Representations},
  author={Wu, Xiaoyang and others},
  booktitle={CVPR},
  year={2025}
}
```

## Further Reading

- **Sonata Paper:** Understanding self-supervised pre-training for point clouds
- **PCA Tutorial:** Interpreting principal component analysis
- **Feature Visualization:** Techniques for high-dimensional feature spaces
- **t-SNE/UMAP:** Alternative dimensionality reduction methods (future work)

## Next Steps

1. **Experiment** with different PCA buffer sizes and downsampling
2. **Compare** PCA visualization with ground truth labels (if available)
3. **Analyze** feature clustering for your specific use case
4. **Extend** the node for your custom downstream tasks
