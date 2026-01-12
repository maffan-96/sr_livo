#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <unordered_set>
#include <mutex>
#include <opencv2/opencv.hpp>

// Forward declarations
class rgbPoint;
struct voxelId;

// Configuration for semantic segmentation
struct SemanticSegmentConfig {
    // SuperVoxel grid resolution (coarser than point voxels)
    double supervoxel_resolution = 0.5;  // meters
    
    // Clustering thresholds
    double normal_similarity_threshold = 0.9;  // cos(angle)
    double color_similarity_threshold = 30.0;  // RGB distance
    double spatial_proximity_threshold = 1.0;  // meters
    
    // Plane detection
    double planarity_threshold = 0.85;  // eigenvalue ratio
    int min_points_for_plane = 10;
    
    // Depth discontinuity detection
    double depth_discontinuity_threshold = 0.3;  // meters
    int depth_check_window_size = 5;  // pixels
    
    // Label transfer
    double label_confidence_threshold = 0.6;
    int min_observations_for_stable_label = 3;
    
    // Edge detection
    double gradient_threshold = 30.0;  // Sobel magnitude
};

// Represents a supervoxel/segment containing multiple points
class SemanticSegment {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    int segment_id;
    
    // Geometry
    Eigen::Vector3d centroid;
    Eigen::Vector3d normal;
    Eigen::Matrix3d covariance;
    double planarity;  // 0-1, how planar is this segment
    
    // Semantics
    int semantic_label;
    int instance_id;
    std::map<int, float> label_histogram;  // label -> weighted vote count
    int total_observations;
    float label_confidence;
    
    // Points in this segment
    std::vector<rgbPoint*> points;
    std::unordered_set<rgbPoint*> point_set;  // For fast lookup
    
    // Neighbor segments (for merging)
    std::unordered_set<int> neighbor_segment_ids;
    
    // State
    bool geometry_dirty;
    bool label_dirty;
    
    SemanticSegment(int id);
    
    void addPoint(rgbPoint* point);
    void removePoint(rgbPoint* point);
    
    void updateGeometry();
    void updateSemanticLabel();
    
    // Weighted label vote (weight based on view quality)
    void addLabelVote(int label, float weight = 1.0f);
    
    bool isEmpty() const { return points.empty(); }
    int numPoints() const { return points.size(); }
    
    // Check if another segment should be merged with this one
    bool shouldMergeWith(const SemanticSegment& other, const SemanticSegmentConfig& config) const;
};

// Hash for supervoxel grid
struct SuperVoxelKey {
    int x, y, z;
    
    SuperVoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    SuperVoxelKey(const Eigen::Vector3d& pos, double resolution) {
        x = static_cast<int>(std::floor(pos.x() / resolution));
        y = static_cast<int>(std::floor(pos.y() / resolution));
        z = static_cast<int>(std::floor(pos.z() / resolution));
    }
    
    bool operator==(const SuperVoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template<> struct hash<SuperVoxelKey> {
        size_t operator()(const SuperVoxelKey& k) const {
            return hash<int>()(k.x) ^ (hash<int>()(k.y) << 1) ^ (hash<int>()(k.z) << 2);
        }
    };
}

// Depth-aware label transfer result
struct LabelTransferResult {
    int semantic_label;
    int instance_id;
    float confidence;
    bool is_at_boundary;
    bool is_depth_discontinuity;
};

// Manager for all semantic segments
class SemanticSegmentManager {
public:
    SemanticSegmentManager();
    ~SemanticSegmentManager();
    
    void setConfig(const SemanticSegmentConfig& config) { config_ = config; }
    
    // Point-segment assignment
    SemanticSegment* getOrCreateSegment(const Eigen::Vector3d& position);
    SemanticSegment* getSegment(int segment_id);
    void assignPointToSegment(rgbPoint* point, const Eigen::Vector3d& position);
    
    // Depth-aware label transfer
    LabelTransferResult computeDepthAwareLabelTransfer(
        const Eigen::Vector3d& point_world,
        const cv::Mat& depth_image,  // or compute from projection
        const cv::Mat& semantic_masks,
        const cv::Mat& rgb_image,
        double u, double v,
        double point_depth,
        const std::vector<std::tuple<cv::Mat, int, int32_t>>& masks
    );
    
    // Image gradient boundary detection
    bool isAtImageBoundary(const cv::Mat& gray_image, double u, double v);
    
    // Segment refinement (call periodically)
    void refinementStep();
    
    // Merge similar adjacent segments
    void mergeSegments(int seg_id_a, int seg_id_b);
    
    // Split segment with inconsistent labels
    void splitInconsistentSegment(int segment_id);
    
    // Plane clustering within a segment
    void detectPlanesInSegment(int segment_id);
    
    // Propagate segment labels to points
    void propagateLabelsToPoints();
    
    // Statistics
    int numSegments() const { return segments_.size(); }
    int numActiveSegments() const;
    
    // Thread safety
    std::shared_ptr<std::mutex> mutex_segments;
    
private:
    SemanticSegmentConfig config_;
    
    // Segment storage
    std::unordered_map<int, SemanticSegment*> segments_;
    int next_segment_id_;
    
    // Spatial index: supervoxel key -> segment id
    std::unordered_map<SuperVoxelKey, int> supervoxel_to_segment_;
    
    // Compute local depth from neighbors for discontinuity check
    double computeLocalMedianDepth(
        const std::vector<std::tuple<cv::Mat, int, int32_t>>& masks,
        double u, double v, int window_size
    );
    
    // Find neighboring supervoxel keys
    std::vector<SuperVoxelKey> getNeighborKeys(const SuperVoxelKey& key);
};