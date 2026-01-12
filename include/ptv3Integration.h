/**
 * @file ptv3Integration.h
 * @brief Integration layer for PTv3 with SR_LIVO semantic fusion pipeline
 *
 * This module bridges PTv3 feature extraction with the existing SR_LIVO
 * semantic mapping system, providing feature-based semantic fusion.
 */

#pragma once

#include <memory>
#include <vector>
#include <mutex>

#include "ptv3FeatureExtractor.h"
#include "rgbMapTracker.h"
#include "cloudMap.h"

// ============================================================================
// Enhanced rgbPoint Methods (add to cloudMap.h rgbPoint class)
// ============================================================================

/**
 * Add these methods to rgbPoint class in cloudMap.h:
 *
 * // PTv3 feature storage
 * private:
 *     Eigen::VectorXf ptv3_feature_;
 *     std::vector<Eigen::VectorXf> feature_history_;
 *     int feature_observation_count_ = 0;
 *
 * public:
 *     void addPTv3Feature(const Eigen::VectorXf& feature, float weight = 1.0f);
 *     Eigen::VectorXf getPTv3Feature() const { return ptv3_feature_; }
 *     bool hasPTv3Feature() const { return feature_observation_count_ > 0; }
 *     int getFeatureObservationCount() const { return feature_observation_count_; }
 */

// ============================================================================
// PTv3-Enhanced Semantic Fusion
// ============================================================================

class PTv3SemanticFusion {
public:
    // ========================================================================
    // Constructor / Destructor
    // ========================================================================

    PTv3SemanticFusion();
    ~PTv3SemanticFusion();

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Initialize PTv3 model and configuration
     */
    bool initialize(const PTv3Config& config);

    /**
     * @brief Check if system is ready
     */
    bool isReady() const { return ptv3_extractor_ && ptv3_extractor_->isReady(); }

    // ========================================================================
    // Feature Extraction Pipeline
    // ========================================================================

    /**
     * @brief Extract PTv3 features for a point cloud frame
     * This is called after cloudProcessing but before adding to map
     */
    bool extractFeaturesForFrame(
        cloudFrame* frame,
        std::vector<point3D>& points_with_features
    );

    /**
     * @brief Extract features for points in voxel during rendering
     * Called from rgbMapTracker::renderPointsInRecentVoxel
     */
    bool extractFeaturesForVoxel(
        const std::vector<rgbPoint*>& voxel_points,
        PTv3Output& output
    );

    // ========================================================================
    // Feature-Based Semantic Transfer
    // ========================================================================

    /**
     * @brief Transfer semantic labels using PTv3 features
     * This replaces/augments direct projection method
     */
    struct FeatureBasedTransfer {
        rgbPoint* point;
        int semantic_label;
        int instance_id;
        float confidence;
        float feature_similarity;
        bool from_3d_prediction;  // True if from PTv3 direct prediction
        bool from_feature_match;  // True if from feature matching
    };

    std::vector<FeatureBasedTransfer> transferSemanticsFeatureBased(
        const std::vector<rgbPoint*>& points,
        cloudFrame* frame,
        const cv::Mat& semantic_mask
    );

    /**
     * @brief Hybrid fusion: combines PTv3 + traditional projection
     * Priority: PTv3 direct > Feature matching > Projection
     */
    void hybridSemanticFusion(
        voxelHashMap& map,
        cloudFrame* frame,
        const std::vector<voxelId>& voxels_to_process
    );

    // ========================================================================
    // Feature Matching
    // ========================================================================

    /**
     * @brief Match point cloud features to semantic mask regions
     */
    struct SemanticMatch {
        int point_idx;
        cv::Point2i best_pixel;
        int semantic_label;
        float match_confidence;
        float feature_similarity;
        Eigen::VectorXf matched_feature;
    };

    std::vector<SemanticMatch> matchPointsToSemanticMask(
        const std::vector<rgbPoint*>& points,
        const PTv3Output& ptv3_output,
        cloudFrame* frame,
        const cv::Mat& semantic_mask
    );

    // ========================================================================
    // Boundary Refinement using Features
    // ========================================================================

    /**
     * @brief Refine semantic labels at boundaries using feature consistency
     */
    void refineBoundaryLabelsWithFeatures(
        voxelHashMap& map,
        const std::vector<voxelId>& boundary_voxels
    );

    /**
     * @brief Check if point is at geometric-semantic boundary
     */
    bool isAtFeatureDiscontinuity(
        rgbPoint* point,
        const std::vector<rgbPoint*>& neighbors,
        float feature_threshold = 0.5f
    );

    // ========================================================================
    // Temporal Feature Consistency
    // ========================================================================

    /**
     * @brief Track feature consistency across frames
     */
    void updateTemporalFeatureConsistency(
        rgbPoint* point,
        const Eigen::VectorXf& new_feature,
        int new_label,
        float confidence
    );

    /**
     * @brief Get refined label based on temporal feature voting
     */
    int getTemporalConsensusLabel(rgbPoint* point);

    // ========================================================================
    // Statistics and Diagnostics
    // ========================================================================

    struct FusionStatistics {
        int total_points_processed = 0;
        int points_from_ptv3_direct = 0;
        int points_from_feature_match = 0;
        int points_from_projection = 0;
        int boundary_points_refined = 0;
        double avg_feature_extraction_time_ms = 0.0;
        double avg_matching_time_ms = 0.0;
    };

    FusionStatistics getStatistics() const { return fusion_stats_; }
    void resetStatistics() { fusion_stats_ = FusionStatistics(); }
    void printStatistics() const;

    // ========================================================================
    // Configuration
    // ========================================================================

    struct FusionConfig {
        // Fusion strategy
        bool use_ptv3_direct_prediction = true;   // Use PTv3's direct semantic prediction
        bool use_feature_matching = true;          // Use feature-based matching
        bool use_traditional_projection = true;    // Fallback to projection

        // Priority weights
        float ptv3_direct_weight = 0.7f;
        float feature_match_weight = 0.2f;
        float projection_weight = 0.1f;

        // Matching thresholds
        float feature_similarity_threshold = 0.6f;  // Minimum feature similarity
        int search_window_pixels = 20;              // Pixel search radius
        float boundary_feature_threshold = 0.5f;    // Feature discontinuity threshold

        // Temporal consistency
        bool enable_temporal_filtering = true;
        int temporal_window_size = 5;              // Frames to consider
        float temporal_consensus_threshold = 0.6f; // Agreement threshold

        // Performance
        bool enable_async_extraction = true;       // Extract features in background
        int batch_size = 4096;                     // Points per PTv3 batch
    };

    void setFusionConfig(const FusionConfig& config) { fusion_config_ = config; }
    FusionConfig getFusionConfig() const { return fusion_config_; }

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * @brief Compute compatibility between 3D feature and 2D semantic region
     */
    float computeFeatureMaskCompatibility(
        const Eigen::VectorXf& point_feature,
        const cv::Mat& semantic_mask,
        cv::Point2i pixel,
        int search_radius
    );

    /**
     * @brief Extract 2D feature context from semantic mask
     */
    Eigen::VectorXf extract2DSemanticContext(
        const cv::Mat& semantic_mask,
        cv::Point2i center,
        int radius
    );

    /**
     * @brief Fuse multiple semantic predictions with confidence weighting
     */
    int fuseSemanticPredictions(
        int ptv3_label,
        float ptv3_conf,
        int matched_label,
        float matched_conf,
        int proj_label,
        float proj_conf
    );

    // ========================================================================
    // Member Variables
    // ========================================================================

    // PTv3 extractor
    std::shared_ptr<PTv3FeatureExtractor> ptv3_extractor_;

    // Configuration
    PTv3Config ptv3_config_;
    FusionConfig fusion_config_;

    // Feature cache (voxel-based)
    struct VoxelFeatureCache {
        PTv3Output features;
        double timestamp;
    };
    std::unordered_map<std::string, VoxelFeatureCache> voxel_feature_cache_;
    std::mutex cache_mutex_;

    // Statistics
    FusionStatistics fusion_stats_;
    std::mutex stats_mutex_;

    // Temporal tracking
    struct PointTemporalHistory {
        std::deque<std::pair<int, float>> label_history;  // (label, confidence)
        std::deque<Eigen::VectorXf> feature_history;
        double last_update_time;
    };
    std::unordered_map<int, PointTemporalHistory> temporal_tracker_;
    std::mutex temporal_mutex_;
};

// ============================================================================
// Modified rgbMapTracker with PTv3 Integration
// ============================================================================

/**
 * @brief Enhanced version of rgbMapTracker::renderPointsInRecentVoxel
 * that uses PTv3 features for semantic fusion
 *
 * Usage:
 * 1. Create PTv3SemanticFusion instance in lioOptimization
 * 2. Call hybridSemanticFusion instead of traditional renderPointsInRecentVoxel
 * 3. Semantic labels will be assigned using PTv3 features + projection
 */
class PTv3EnhancedRenderer {
public:
    PTv3EnhancedRenderer(
        std::shared_ptr<PTv3SemanticFusion> ptv3_fusion,
        rgbMapTracker* rgb_tracker
    );

    /**
     * @brief Render points with PTv3-enhanced semantic fusion
     * Drop-in replacement for rgbMapTracker::renderPointsInRecentVoxel
     */
    void renderWithFeatures(
        voxelHashMap& map,
        cloudFrame* frame,
        const std::vector<voxelId>& voxels_for_render,
        const double& obs_time
    );

private:
    std::shared_ptr<PTv3SemanticFusion> ptv3_fusion_;
    rgbMapTracker* rgb_tracker_;
};

// ============================================================================
// Integration Helper Functions
// ============================================================================

/**
 * @brief Add PTv3 feature to rgbPoint (implementation for cloudMap.h)
 */
inline void addPTv3FeatureToPoint(
    rgbPoint* point,
    const Eigen::VectorXf& feature,
    float weight = 1.0f
) {
    // This function should be added to rgbPoint class implementation
    // For now, we use external storage
    // TODO: Move to rgbPoint::addPTv3Feature() method
}

/**
 * @brief Get PTv3 feature from rgbPoint
 */
inline Eigen::VectorXf getPTv3FeatureFromPoint(const rgbPoint* point) {
    // TODO: Implement rgbPoint::getPTv3Feature() method
    return Eigen::VectorXf::Zero(256);
}

/**
 * @brief Convert PTv3Output to map of point features
 */
std::unordered_map<int, Eigen::VectorXf> createFeatureMap(
    const PTv3Output& output,
    const std::vector<rgbPoint*>& points
);

/**
 * @brief Compute feature centroid for a voxel
 */
Eigen::VectorXf computeVoxelFeatureCentroid(
    const std::vector<rgbPoint*>& voxel_points
);

/**
 * @brief Check feature consistency in neighborhood
 */
bool checkFeatureConsistency(
    const Eigen::VectorXf& query_feature,
    const std::vector<Eigen::VectorXf>& neighbor_features,
    float threshold = 0.6f
);
