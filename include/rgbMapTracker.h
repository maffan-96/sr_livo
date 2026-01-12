/**
 * @file rgbMapTracker.h
 * @brief RGB Map Tracker with Geometric Segmentation for Semantic Label Refinement
 * 
 * This header integrates with the existing SR-LIVO codebase.
 * It does NOT redefine types already in cloudMap.h
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>

// Include existing cloudMap.h for type definitions
// (voxelId, voxel, voxelBlock, voxelHashMap, rgbPoint, cloudFrame are defined there)
#include "cloudMap.h"

// Forward declarations
class GeometricLabelManager;
struct GeometricSegmentationConfig;
class cloudFrame;  // Forward declare if not fully defined in cloudMap.h

// ============================================================================
// Label Transfer Information (for geometric-aware transfer)
// ============================================================================

struct LabelTransferInfo {
    int label;                      // Semantic label (0 = unknown)
    int instance_id;                // Instance ID
    float confidence;               // Transfer confidence [0, 1]
    bool is_boundary;               // True if at image edge/gradient boundary
    bool is_depth_discontinuity;    // True if at depth discontinuity
    
    LabelTransferInfo() 
        : label(0)
        , instance_id(0)
        , confidence(0.0f)
        , is_boundary(false)
        , is_depth_discontinuity(false) {}
};

// ============================================================================
// RGB Map Tracker Class
// ============================================================================

class rgbMapTracker {
public:
    // ========================================================================
    // Constructor / Destructor
    // ========================================================================
    rgbMapTracker();
    ~rgbMapTracker();
    
    // ========================================================================
    // PUBLIC MEMBERS (needed by existing SR-LIVO code)
    // ========================================================================
    
    // Point storage - accessed by lioOptimization
    std::vector<rgbPoint*> rgb_points_vec;
    
    // Voxel tracking - accessed by lioOptimization, imageProcessing
    std::vector<voxelId> voxels_recent_visited;
    int number_of_new_visited_voxel;
    
    // Frame tracking - accessed by lioOptimization
    int updated_frame_index;
    
    // Projection parameters - accessed by imageProcessing
    double minimum_depth_for_projection;
    double maximum_depth_for_projection;
    
    // Points for projection - accessed by opticalFlowTracker
    std::vector<rgbPoint*>* points_rgb_vec_for_projection;
    std::vector<cv::Point2f>* points_2d_vec_for_projection;
    
    // Synchronization - accessed by lioOptimization (MUST be public)
    std::shared_ptr<std::mutex> mutex_rgb_points_vec;
    std::shared_ptr<std::mutex> mutex_frame_index;
    
    // ========================================================================
    // EXISTING METHODS (from original SR-LIVO)
    // ========================================================================
    
    void selectPointsForProjection(
        voxelHashMap& map,
        cloudFrame* p_frame,
        std::vector<rgbPoint*>* rgb_points_vec,
        std::vector<cv::Point2f>* points_2d_vec,
        int window_size,
        int skip
    );
    
    void updatePoseForProjection(cloudFrame* p_frame, double fov_margin);
    
    void refreshPointsForProjection(voxelHashMap& map);
    
    void renderPointsInRecentVoxel(
        voxelHashMap& map,
        cloudFrame* p_frame,
        std::vector<voxelId>* voxels_for_render,
        const double& obs_time
    );
    
    void threadRenderPointsInVoxel(
        voxelHashMap& map,
        const int& voxel_start,
        const int& voxel_end,
        cloudFrame* p_frame,
        const std::vector<voxelId>* voxels_for_render,
        const double obs_time
    );
    
    // ========================================================================
    // NEW: Geometric Segmentation Configuration
    // ========================================================================
    
    void setGeometricSegmentationEnabled(bool enabled) { 
        geometric_segmentation_enabled_ = enabled; 
    }
    bool isGeometricSegmentationEnabled() const { 
        return geometric_segmentation_enabled_; 
    }
    
    void setGeometricSegmentationConfig(const GeometricSegmentationConfig& config);
    
    void setDepthDiscontinuityThreshold(double threshold) {
        depth_discontinuity_threshold_ = threshold;
    }
    
    void setGradientThreshold(double threshold) {
        gradient_threshold_ = threshold;
    }
    
    void setRefinementInterval(int frames) {
        refinement_frame_interval_ = frames;
    }
    
    // ========================================================================
    // NEW: Semantic Label Refinement
    // ========================================================================
    
    /**
     * @brief Refine semantic labels using geometric segmentation
     * @param full_resegmentation If true, perform full resegmentation
     */
    void refineSemanticLabels(bool full_resegmentation = false);
    
    /**
     * @brief Apply refined labels from geometric segments back to points
     * @param map The voxel hash map to update
     */
    void applyRefinedLabelsToMap(voxelHashMap& map);
    
    // ========================================================================
    // NEW: Export and Visualization
    // ========================================================================
    
    void exportSemanticPointCloud(voxelHashMap& map, const std::string& filename);
    
    void getSegmentationVisualization(
        voxelHashMap& map,
        std::vector<Eigen::Vector3d>& positions,
        std::vector<Eigen::Vector3i>& colors
    );
    
    void getBoundaryVisualization(
        voxelHashMap& map,
        std::vector<Eigen::Vector3d>& boundary_points
    );
    
    // ========================================================================
    // NEW: Statistics
    // ========================================================================
    
    void printStatistics() const;
    int getNumSegments() const;
    int getRenderPointCount() const { return render_point_count_; }
    
    // ========================================================================
    // Existing public parameters
    // ========================================================================
    
    double image_obs_cov;
    double image_obs_threshold;
    int minimum_pts;
    int grid_size;

private:
    // ========================================================================
    // NEW: Geometric-Aware Rendering
    // ========================================================================
    
    void threadRenderPointsInVoxelGeometric(
        voxelHashMap& map,
        const int& voxel_start,
        const int& voxel_end,
        cloudFrame* p_frame,
        const std::vector<voxelId>* voxels_for_render,
        const double obs_time,
        const cv::Mat& gray_image
    );
    
    LabelTransferInfo computeGeometricAwareLabelTransfer(
        const Eigen::Vector3d& point_world,
        cloudFrame* p_frame,
        double u, double v,
        int u_i, int v_i,
        double point_depth,
        const cv::Mat& gray_image
    );
    
    bool isAtImageBoundary(const cv::Mat& gray_image, int u, int v);
    bool checkDepthDiscontinuity(cloudFrame* p_frame, int u, int v, double point_depth);
    float computeMaskConfidence(const cv::Mat& mask, int u, int v);
    
    // ========================================================================
    // Private State
    // ========================================================================
    
    // Geometric segmentation
    GeometricLabelManager* geometric_label_manager_;
    bool geometric_segmentation_enabled_;
    
    double depth_discontinuity_threshold_;
    double gradient_threshold_;
    
    int refinement_frame_interval_;
    int refinement_counter_;
    int full_resegmentation_interval_;
    int full_resegmentation_counter_;
    
    // Points pending segmentation
    std::vector<rgbPoint*> pending_points_for_segmentation_;
    std::vector<rgbPoint*> frame_pending_points_;
    std::mutex mutex_pending_points_;
    
    // Statistics
    int render_point_count_;
    int frame_index_;
    double total_render_time_ms_;
    double total_refinement_time_ms_;
    int render_call_count_;
    
    // Voxels for current render pass
    std::vector<voxelId> g_voxel_for_render_;
};