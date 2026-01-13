#!/usr/bin/env python3
"""
Sonata ROS Bridge Node
Performs 3D semantic segmentation on point clouds using Sonata (PTv3)
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


class SonataSegmentationNode:
    """ROS node for Sonata-based 3D semantic segmentation"""

    def __init__(self):
        rospy.init_node('sonata_segmentation_node', anonymous=False)

        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/cloud_registered_current')
        self.output_topic = rospy.get_param('~output_topic', '/sonata/semantic_cloud')
        self.model_name = rospy.get_param('~model_name', 'facebook/sonata')
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.downsample_voxel = rospy.get_param('~downsample_voxel', 0.02)  # 2cm voxel downsampling
        self.use_normals = rospy.get_param('~use_normals', True)
        self.use_colors = rospy.get_param('~use_colors', True)
        self.batch_size = rospy.get_param('~batch_size', 1)
        self.num_classes = rospy.get_param('~num_classes', 20)  # ScanNet default
        self.inference_mode = rospy.get_param('~inference_mode', 'feature')  # 'feature' or 'segmentation'

        # Statistics
        self.msg_count = 0
        self.total_inference_time = 0.0
        self.total_points_processed = 0

        # Load model
        self.model = None
        if SONATA_AVAILABLE:
            self.load_model()
        else:
            rospy.logwarn("Running in dummy mode without Sonata model")

        # Publishers and Subscribers
        self.pub_semantic = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.pub_features = rospy.Publisher(self.output_topic + '_features', PointCloud2, queue_size=1)

        self.sub_cloud = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1
        )

        rospy.loginfo(f"Sonata Segmentation Node initialized")
        rospy.loginfo(f"  Input topic: {self.input_topic}")
        rospy.loginfo(f"  Output topic: {self.output_topic}")
        rospy.loginfo(f"  Device: {self.device}")
        rospy.loginfo(f"  Model: {self.model_name}")
        rospy.loginfo(f"  Inference mode: {self.inference_mode}")

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
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA

        normals = np.zeros_like(coords)

        try:
            # Find k nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(coords)
            _, indices = nbrs.kneighbors(coords)

            # Estimate normal for each point
            for i in range(len(coords)):
                neighbors = coords[indices[i]]
                pca = PCA(n_components=3)
                pca.fit(neighbors)
                # Normal is the eigenvector with smallest eigenvalue
                normals[i] = pca.components_[2]
        except Exception as e:
            rospy.logwarn(f"Normal estimation failed: {e}, using zero normals")
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

    def run_inference(self, coords, colors, normals):
        """Run Sonata inference on point cloud"""
        if self.model is None:
            # Return dummy predictions
            return np.random.randint(0, self.num_classes, size=len(coords))

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

            # Run inference
            with torch.no_grad():
                output = self.model(data_dict)

            if self.inference_mode == 'segmentation' and 'seg_logits' in output:
                # Get semantic predictions
                predictions = output['seg_logits'].argmax(dim=-1).cpu().numpy()
            else:
                # Return features or dummy predictions
                rospy.logwarn_once("Model output does not contain 'seg_logits', returning random labels")
                predictions = np.random.randint(0, self.num_classes, size=len(coords))

            return predictions

        except Exception as e:
            rospy.logerr(f"Inference failed: {e}")
            return np.zeros(len(coords), dtype=np.int32)

    def numpy_to_pointcloud2(self, coords, colors, labels, frame_id, stamp):
        """Convert numpy arrays to ROS PointCloud2 with semantic labels"""
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.UINT8, 1),
            PointField('g', 13, PointField.UINT8, 1),
            PointField('b', 14, PointField.UINT8, 1),
            PointField('label', 15, PointField.UINT8, 1),
        ]

        # Create point cloud data
        points = []
        for i in range(len(coords)):
            x, y, z = coords[i]
            r, g, b = (colors[i] * 255).astype(np.uint8) if colors is not None else [255, 255, 255]
            label = int(labels[i]) % 256  # Ensure it fits in uint8

            # Pack data
            point = struct.pack('fffBBBB', x, y, z, r, g, b, label)
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

            # Run inference on downsampled cloud
            labels_ds = self.run_inference(coords_ds, colors_ds, normals_ds)

            # Upsample labels back to original resolution
            labels = labels_ds[voxel_map]

            # Convert back to ROS message
            semantic_cloud = self.numpy_to_pointcloud2(
                coords, colors, labels, msg.header.frame_id, msg.header.stamp
            )

            # Publish
            self.pub_semantic.publish(semantic_cloud)

            # Statistics
            inference_time = time.time() - start_time
            self.msg_count += 1
            self.total_inference_time += inference_time
            self.total_points_processed += original_num_points

            if self.msg_count % 10 == 0:
                avg_time = self.total_inference_time / self.msg_count
                avg_points = self.total_points_processed / self.msg_count
                rospy.loginfo(f"Stats: {self.msg_count} clouds, avg {avg_time:.3f}s/cloud, {avg_points:.0f} points/cloud")

        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main loop"""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = SonataSegmentationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
