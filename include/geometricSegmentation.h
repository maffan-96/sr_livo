/**
 * @file geometricSegmentation.h
 * @brief Geometric Point Cloud Segmentation for Semantic Label Refinement
 * 
 * This module provides geometry-driven segmentation that prevents label bleeding
 * by detecting object boundaries using:
 * - Surface normal discontinuities
 * - Curvature-based edge detection
 * - Convexity analysis for junction detection
 * 
 * The key insight is that geometric discontinuities (normal changes, convexity breaks)
 * define where objects actually separate, regardless of semantic class.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <atomic>

// Forward declarations
class rgbPoint;

// ============================================================================
// Configuration
// ============================================================================

struct GeometricSegmentationConfig {
    // Neighborhood search parameters
    double search_radius = 0.12;              // meters, for normal estimation
    int min_neighbors_for_normal = 5;         // minimum neighbors for valid normal
    int max_neighbors = 30;                   // maximum neighbors to consider
    
    // Geometric feature thresholds
    double normal_threshold = 0.85;           // cos(angle) ~ 32 degrees
    double curvature_threshold = 0.08;        // normalized curvature [0,1]
    double convexity_threshold = -0.015;      // negative = concave boundary
    
    // Region growing parameters
    double region_grow_radius = 0.08;         // meters, for region expansion
    int min_segment_size = 8;                 // minimum points per segment
    int max_segment_size = 50000;             // maximum points per segment
    
    // Label assignment parameters
    double label_majority_threshold = 0.4;    // minimum fraction for label
    int min_labeled_points = 3;               // minimum labeled points needed
    float min_confidence_for_propagation = 0.3f;  // minimum confidence to propagate
    
    // Performance parameters
    bool use_parallel = true;                 // use OpenMP parallelization
    int num_threads = 4;                      // number of threads if parallel
    
    GeometricSegmentationConfig() = default;
};

// ============================================================================
// Per-Point Geometric Features
// ============================================================================

struct PointGeometricFeatures {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Eigen::Vector3d normal;
    double curvature;               // 0 = flat, higher = more curved
    double convexity;               // positive = convex, negative = concave
    bool is_boundary;               // detected as geometric boundary
    bool features_valid;            // features successfully computed
    
    PointGeometricFeatures() 
        : normal(Eigen::Vector3d::UnitZ())
        , curvature(0.0)
        , convexity(0.0)
        , is_boundary(false)
        , features_valid(false) {}
};

// ============================================================================
// Geometric Segment
// ============================================================================

class GeometricSegment {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    int segment_id;
    
    // Points in this segment (indices into point array)
    std::vector<int> point_indices;
    
    // Geometry summary
    Eigen::Vector3d centroid;
    Eigen::Vector3d mean_normal;
    double mean_curvature;
    double planarity;               // 0 = not planar, 1 = perfectly planar
    
    // Bounding box
    Eigen::Vector3d bbox_min;
    Eigen::Vector3d bbox_max;
    
    // Semantic label (assigned from 2D projections)
    int semantic_label;
    int instance_id;
    std::unordered_map<int, float> label_votes;  // label -> weighted count
    float label_confidence;
    int total_vote_count;
    
    // Neighbor segments
    std::unordered_set<int> neighbor_segment_ids;
    
    // State flags
    bool geometry_dirty;
    bool label_dirty;
    
    // Constructors
    GeometricSegment();  // Default constructor required by std::unordered_map
    GeometricSegment(int id);
    
    // Methods
    void computeSummary(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<PointGeometricFeatures>& features
    );
    
    void addLabelVote(int label, float weight = 1.0f, int instance = 0);
    void assignLabel(const GeometricSegmentationConfig& config);
    void clearVotes();
    
    bool isEmpty() const { return point_indices.empty(); }
    int numPoints() const { return static_cast<int>(point_indices.size()); }
};

// ============================================================================
// KD-Tree for Spatial Queries
// ============================================================================

// Simple KD-Tree implementation for neighbor search
class SimpleKDTree {
public:
    SimpleKDTree();
    ~SimpleKDTree();
    
    void build(const std::vector<Eigen::Vector3d>& points);
    void clear();
    
    std::vector<int> radiusSearch(
        const Eigen::Vector3d& query, 
        double radius
    ) const;
    
    std::vector<int> knnSearch(
        const Eigen::Vector3d& query,
        int k
    ) const;
    
    bool isBuilt() const { return is_built_; }
    
private:
    struct Node {
        int point_idx;
        int split_dim;
        double split_val;
        Node* left;
        Node* right;
        
        Node() : point_idx(-1), split_dim(0), split_val(0), 
                 left(nullptr), right(nullptr) {}
    };
    
    Node* root_;
    const std::vector<Eigen::Vector3d>* points_;
    bool is_built_;
    
    Node* buildRecursive(std::vector<int>& indices, int depth);
    void searchRadius(
        Node* node, 
        const Eigen::Vector3d& query, 
        double radius_sq,
        std::vector<int>& results
    ) const;
    void destroyTree(Node* node);
};

// ============================================================================
// Main Geometric Segmenter Class
// ============================================================================

class GeometricSegmenter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    GeometricSegmenter();
    ~GeometricSegmenter();
    
    // Configuration
    void setConfig(const GeometricSegmentationConfig& config) { config_ = config; }
    const GeometricSegmentationConfig& getConfig() const { return config_; }
    
    // ========================================================================
    // Main Interface
    // ========================================================================
    
    /**
     * @brief Set point cloud for segmentation
     * @param positions Point positions
     */
    void setPoints(const std::vector<Eigen::Vector3d>& positions);
    
    /**
     * @brief Compute geometric features (normals, curvature, convexity)
     */
    void computeFeatures();
    
    /**
     * @brief Perform region growing segmentation
     */
    void performSegmentation();
    
    /**
     * @brief Full pipeline: compute features + segment
     */
    void process() {
        computeFeatures();
        performSegmentation();
    }
    
    // ========================================================================
    // Results Access
    // ========================================================================
    
    int getSegmentId(int point_index) const;
    const GeometricSegment* getSegment(int segment_id) const;
    GeometricSegment* getSegmentMutable(int segment_id);
    
    const std::vector<PointGeometricFeatures>& getFeatures() const { return features_; }
    const PointGeometricFeatures& getFeature(int point_index) const { return features_[point_index]; }
    
    int numPoints() const { return static_cast<int>(points_.size()); }
    int numSegments() const { return static_cast<int>(segments_.size()); }
    
    std::vector<int> getAllSegmentIds() const;
    
    // ========================================================================
    // Label Management
    // ========================================================================
    
    /**
     * @brief Add label vote for a specific point
     */
    void addLabelVoteToPoint(int point_index, int label, float weight = 1.0f, int instance = 0);
    
    /**
     * @brief Propagate point-level votes to segment level
     */
    void propagateLabelsToSegments();
    
    /**
     * @brief Get final labels for all points (from their segments)
     */
    void propagateSegmentLabelsToPoints(
        std::vector<int>& out_labels, 
        std::vector<int>& out_instances
    );
    
    /**
     * @brief Clear all label votes
     */
    void clearAllLabels();
    
    // ========================================================================
    // Incremental Updates
    // ========================================================================
    
    /**
     * @brief Add new points incrementally
     * @param new_positions New point positions
     * @return Starting index of new points
     */
    int addPointsIncremental(const std::vector<Eigen::Vector3d>& new_positions);
    
    /**
     * @brief Update segmentation incrementally for new points
     */
    void updateSegmentationIncremental();
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    void printStatistics() const;
    
private:
    GeometricSegmentationConfig config_;
    
    // Point data
    std::vector<Eigen::Vector3d> points_;
    std::unique_ptr<SimpleKDTree> kdtree_;
    std::vector<PointGeometricFeatures> features_;
    
    // Per-point label votes
    std::vector<std::unordered_map<int, float>> point_label_votes_;
    std::vector<int> point_instance_ids_;
    
    // Segmentation results
    std::vector<int> point_to_segment_;  // point index -> segment id
    std::unordered_map<int, GeometricSegment> segments_;
    int next_segment_id_;
    
    // State
    bool features_computed_;
    bool segmentation_done_;
    int last_segmented_point_count_;
    
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    void computeNormalAndCurvature(
        int point_idx, 
        const std::vector<int>& neighbor_indices
    );
    
    void computeConvexity(
        int point_idx, 
        const std::vector<int>& neighbor_indices
    );
    
    void detectBoundaryPoints();
    
    void regionGrow();
    
    bool canGrowTo(int from_idx, int to_idx) const;
    
    void handleUnassignedPoints();
    
    void updateSegmentNeighbors();
};

// ============================================================================
// GeometricLabelManager - Integration with rgbMapTracker
// ============================================================================

class GeometricLabelManager {
public:
    GeometricLabelManager();
    ~GeometricLabelManager();
    
    // Configuration
    void setConfig(const GeometricSegmentationConfig& config);
    const GeometricSegmentationConfig& getConfig() const { return config_; }
    
    // ========================================================================
    // Point Registration
    // ========================================================================
    
    /**
     * @brief Register new points for segmentation
     * @param points Vector of point pointers
     */
    void onPointsAdded(const std::vector<rgbPoint*>& points);
    
    // ========================================================================
    // Label Input
    // ========================================================================
    
    /**
     * @brief Record a label projection from 2D to a point
     * @param point Pointer to the point
     * @param label Semantic label
     * @param instance Instance ID
     * @param confidence Projection confidence [0,1]
     */
    void onLabelProjected(
        rgbPoint* point, 
        int label, 
        int instance, 
        float confidence
    );
    
    // ========================================================================
    // Processing
    // ========================================================================
    
    /**
     * @brief Incremental update (for periodic refinement)
     */
    void update();
    
    /**
     * @brief Full resegmentation (more expensive, more accurate)
     */
    void fullUpdate();
    
    // ========================================================================
    // Label Output
    // ========================================================================
    
    /**
     * @brief Get refined label for a point
     * @param point Pointer to the point
     * @param out_label Output semantic label
     * @param out_instance Output instance ID
     * @return True if a refined label is available
     */
    bool getRefinedLabel(
        rgbPoint* point, 
        int& out_label, 
        int& out_instance
    );
    
    /**
     * @brief Get segment ID for a point
     * @param point Pointer to the point
     * @return Segment ID or -1 if not segmented
     */
    int getSegmentId(rgbPoint* point);
    
    /**
     * @brief Check if point is at geometric boundary
     * @param point Pointer to the point
     * @return True if point is a boundary point
     */
    bool isBoundaryPoint(rgbPoint* point);
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    int getNumSegments() const;
    int getNumPoints() const;
    void printStatistics() const;
    
    // ========================================================================
    // Reset
    // ========================================================================
    
    void reset();
    
private:
    GeometricSegmenter segmenter_;
    GeometricSegmentationConfig config_;
    
    // Mapping between rgbPoint* and internal indices
    std::unordered_map<rgbPoint*, int> point_to_index_;
    std::vector<rgbPoint*> index_to_point_;
    
    // Pending updates
    std::vector<rgbPoint*> pending_points_;
    bool needs_rebuild_;
    
    // Cached refined labels (for fast lookup)
    std::unordered_map<rgbPoint*, std::pair<int, int>> cached_labels_;
    bool cache_valid_;
    
    // Thread safety
    std::mutex mutex_;
    
    // Internal methods
    void rebuildSegmenter();
    void updateCache();
};