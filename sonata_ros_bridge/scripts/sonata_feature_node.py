#!/usr/bin/env python3
"""
Sonata ROS Bridge Node
Extracts features from point clouds using Sonata (PTv3) and visualizes them using PCA
"""

import rospy
import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import struct
import time

try:
    import sonata
    import sonata.model
    SONATA_AVAILABLE = True
except ImportError:
    rospy.logwarn("Sonata not available. Install sonata package for inference.")
    SONATA_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    rospy.logwarn("scikit-learn not available. Install for PCA visualization.")
    SKLEARN_AVAILABLE = False


class SonataFeatureNode:
    """ROS node for Sonata-based feature extraction and PCA visualization"""

    def __init__(self):
        rospy.init_node('sonata_feature_node', anonymous=False)

        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/cloud_registered_current')
        self.output_topic = rospy.get_param('~output_topic', '/sonata/feature_cloud')
        self.output_pca_topic = rospy.get_param('~output_pca_topic', '/sonata/pca_cloud')
        self.model_name = rospy.get_param('~model_name', 'facebook/sonata')
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.downsample_voxel = rospy.get_param('~downsample_voxel', 0.02)  # 2cm voxel downsampling
        self.use_normals = rospy.get_param('~use_normals', True)
        self.use_colors = rospy.get_param('~use_colors', True)
        self.pca_components = rospy.get_param('~pca_components', 3)  # Number of PCA components
        self.normalize_features = rospy.get_param('~normalize_features', True)
        self.feature_layer = rospy.get_param('~feature_layer', 'final')  # 'final' or layer index

        # Statistics
        self.msg_count = 0
        self.total_inference_time = 0.0
        self.total_points_processed = 0

        # PCA model (fitted incrementally)
        self.pca = None
        self.scaler = None
        self.features_buffer = []
        self.buffer_size = rospy.get_param('~pca_buffer_size', 10)  # Number of frames to buffer for PCA

        # Load model
        self.model = None
        if SONATA_AVAILABLE:
            self.load_model()
        else:
            rospy.logwarn("Running in dummy mode without Sonata model")

        # Publishers and Subscribers
        self.pub_features = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.pub_pca = rospy.Publisher(self.output_pca_topic, PointCloud2, queue_size=1)

        self.sub_cloud = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1
        )

        rospy.loginfo(f"Sonata Feature Extraction Node initialized")
        rospy.loginfo(f"  Input topic: {self.input_topic}")
        rospy.loginfo(f"  Output feature topic: {self.output_topic}")
        rospy.loginfo(f"  Output PCA topic: {self.output_pca_topic}")
        rospy.loginfo(f"  Device: {self.device}")
        rospy.loginfo(f"  Model: {self.model_name}")
        rospy.loginfo(f"  PCA components: {self.pca_components}")
        rospy.loginfo(f"  PCA buffer size: {self.buffer_size} frames")

    def load_model(self):
        """Load Sonata pre-trained model"""
        try:
            rospy.loginfo(f"Loading Sonata model from {self.model_name}...")
            self.model = sonata.model.load("sonata", repo_id=self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            rospy.loginfo("Sonata model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load Sonata model: {e}")
            self.model = None

    def pointcloud2_to_numpy(self, cloud_msg):
        """Convert ROS PointCloud2 message to numpy arrays"""
        # Read point cloud data
        points_list = []

        for point in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append(point)

        if len(points_list) == 0:
            return None, None, None

        points = np.array(points_list)

        # Extract coordinates (always first 3 fields)
        coords = points[:, :3].astype(np.float32)

        # Extract colors if available (assume RGB or intensity)
        colors = None
        if self.use_colors and points.shape[1] >= 6:
            # Assume fields are x, y, z, intensity/r, g, b or similar
            colors = points[:, 3:6].astype(np.float32)
            # Normalize to [0, 1] if needed
            if colors.max() > 1.0:
                colors = colors / 255.0
        elif self.use_colors and points.shape[1] >= 4:
            # Single intensity channel, replicate to RGB
            intensity = points[:, 3:4].astype(np.float32)
            if intensity.max() > 1.0:
                intensity = intensity / 255.0
            colors = np.tile(intensity, (1, 3))
        else:
            # Default white color
            colors = np.ones((coords.shape[0], 3), dtype=np.float32)

        # Compute normals (simple approach - could be improved)
        normals = None
        if self.use_normals:
            normals = self.estimate_normals(coords)
        else:
            normals = np.zeros((coords.shape[0], 3), dtype=np.float32)

        return coords, colors, normals

    def estimate_normals(self, coords, k=10):
        """Simple normal estimation using PCA"""
        if not SKLEARN_AVAILABLE:
            return np.zeros_like(coords)

        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA

        normals = np.zeros_like(coords)

        try:
            # Find k nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=min(k, len(coords)), algorithm='kd_tree').fit(coords)
            _, indices = nbrs.kneighbors(coords)

            # Estimate normal for each point
            for i in range(len(coords)):
                neighbors = coords[indices[i]]
                pca = PCA(n_components=3)
                pca.fit(neighbors)
                # Normal is the eigenvector with smallest eigenvalue
                normals[i] = pca.components_[2]
        except Exception as e:
            rospy.logwarn_once(f"Normal estimation failed: {e}, using zero normals")
            normals = np.zeros_like(coords)

        return normals.astype(np.float32)

    def voxel_downsample(self, coords, colors, normals, voxel_size):
        """Voxel-based downsampling"""
        if voxel_size <= 0:
            return coords, colors, normals, np.arange(len(coords))

        # Compute voxel indices
        voxel_coords = np.floor(coords / voxel_size).astype(np.int32)

        # Find unique voxels
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)

        # Create index mapping for upsampling later
        voxel_map = np.zeros(len(coords), dtype=np.int32)
        for new_idx, old_idx in enumerate(unique_indices):
            # Find all points in same voxel
            mask = np.all(voxel_coords == voxel_coords[old_idx], axis=1)
            voxel_map[mask] = new_idx

        return (coords[unique_indices],
                colors[unique_indices],
                normals[unique_indices],
                voxel_map)

    def extract_features(self, coords, colors, normals):
        """Extract features using Sonata model"""
        if self.model is None:
            # Return dummy features
            return np.random.randn(len(coords), 256).astype(np.float32)

        try:
            # Prepare input data
            data_dict = {
                'coord': coords,
                'color': colors,
                'normal': normals,
            }

            # Move to device
            for key in data_dict:
                data_dict[key] = torch.from_numpy(data_dict[key]).to(self.device)

            # Add batch dimension if needed
            if 'batch' not in data_dict:
                data_dict['batch'] = torch.zeros(len(coords), dtype=torch.long).to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(data_dict)

            # Extract features from output
            if isinstance(output, dict):
                # Try to get final features
                if 'feat' in output:
                    features = output['feat']
                elif 'features' in output:
                    features = output['features']
                elif 'x' in output:
                    features = output['x']
                else:
                    # Use first tensor value in dict
                    features = list(output.values())[0]
            else:
                features = output

            # Convert to numpy
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

            # Ensure 2D array [N, D]
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)

            return features.astype(np.float32)

        except Exception as e:
            rospy.logerr(f"Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return np.random.randn(len(coords), 256).astype(np.float32)

    def apply_pca(self, features):
        """Apply PCA to features for visualization"""
        if not SKLEARN_AVAILABLE:
            rospy.logwarn_once("scikit-learn not available, skipping PCA")
            return None

        try:
            # Normalize features if requested
            if self.normalize_features:
                if self.scaler is None:
                    self.scaler = StandardScaler()
                    features_scaled = self.scaler.fit_transform(features)
                else:
                    features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

            # Fit or update PCA
            if self.pca is None or len(self.features_buffer) < self.buffer_size:
                # Buffer features for better PCA estimation
                self.features_buffer.append(features_scaled)

                if len(self.features_buffer) >= self.buffer_size:
                    # Concatenate buffered features
                    all_features = np.vstack(self.features_buffer)

                    # Fit PCA
                    self.pca = PCA(n_components=self.pca_components)
                    self.pca.fit(all_features)

                    rospy.loginfo(f"PCA fitted on {len(all_features)} points")
                    rospy.loginfo(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")

                    # Keep only recent features in buffer
                    self.features_buffer = self.features_buffer[-self.buffer_size//2:]

            # Transform features
            if self.pca is not None:
                pca_features = self.pca.transform(features_scaled)

                # Normalize to [0, 1] for visualization
                pca_min = pca_features.min(axis=0)
                pca_max = pca_features.max(axis=0)
                pca_range = pca_max - pca_min
                pca_range[pca_range == 0] = 1  # Avoid division by zero
                pca_normalized = (pca_features - pca_min) / pca_range

                return pca_normalized
            else:
                return None

        except Exception as e:
            rospy.logerr(f"PCA failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def numpy_to_pointcloud2_rgb(self, coords, colors, frame_id, stamp):
        """Convert numpy arrays to ROS PointCloud2 with RGB colors"""
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]

        # Create point cloud data
        points = []
        for i in range(len(coords)):
            x, y, z = coords[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)

            # Pack RGB into float32
            rgb_packed = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]

            # Pack data
            point = struct.pack('ffff', x, y, z, rgb_packed)
            points.append(point)

        # Create PointCloud2 message
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.data = b''.join(points)

        return cloud_msg

    def cloud_callback(self, msg):
        """Process incoming point cloud messages"""
        start_time = time.time()

        try:
            # Convert ROS message to numpy
            coords, colors, normals = self.pointcloud2_to_numpy(msg)

            if coords is None:
                rospy.logwarn("Received empty point cloud")
                return

            original_num_points = len(coords)

            # Downsample for efficiency
            coords_ds, colors_ds, normals_ds, voxel_map = self.voxel_downsample(
                coords, colors, normals, self.downsample_voxel
            )

            rospy.loginfo(f"Point cloud: {original_num_points} -> {len(coords_ds)} points (voxel={self.downsample_voxel}m)")

            # Extract features
            features_ds = self.extract_features(coords_ds, colors_ds, normals_ds)
            rospy.loginfo(f"Extracted features shape: {features_ds.shape}")

            # Apply PCA for visualization
            pca_features = self.apply_pca(features_ds)

            if pca_features is not None:
                # Upsample PCA features to original resolution
                pca_features_full = pca_features[voxel_map]

                # Use first 3 PCA components as RGB
                pca_colors = pca_features_full[:, :3] if pca_features_full.shape[1] >= 3 else np.tile(pca_features_full[:, 0:1], (1, 3))

                # Publish PCA-colored point cloud
                pca_cloud = self.numpy_to_pointcloud2_rgb(
                    coords, pca_colors, msg.header.frame_id, msg.header.stamp
                )
                self.pub_pca.publish(pca_cloud)

            # Publish original point cloud with features (for debugging)
            feature_cloud = self.numpy_to_pointcloud2_rgb(
                coords, colors, msg.header.frame_id, msg.header.stamp
            )
            self.pub_features.publish(feature_cloud)

            # Statistics
            inference_time = time.time() - start_time
            self.msg_count += 1
            self.total_inference_time += inference_time
            self.total_points_processed += original_num_points

            if self.msg_count % 10 == 0:
                avg_time = self.total_inference_time / self.msg_count
                avg_points = self.total_points_processed / self.msg_count
                rospy.loginfo(f"Stats: {self.msg_count} clouds, avg {avg_time:.3f}s/cloud, {avg_points:.0f} points/cloud")
                if self.pca is not None:
                    rospy.loginfo(f"PCA explained variance: {self.pca.explained_variance_ratio_}")

        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main loop"""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = SonataFeatureNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
