/**
 * @file rgbMapTracker.cpp
 * @brief RGB Map Tracker with Geometric Segmentation
 * 
 * This integrates with the existing SR-LIVO codebase while adding
 * geometric segmentation for improved semantic label transfer.
 */

// IMPORTANT: Include lioOptimization.h FIRST to get full cloudFrame definition
// before rgbMapTracker.h's forward declaration is seen
#include "lioOptimization.h"
#include "rgbMapTracker.h"
#include "geometricSegmentation.h"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <tuple>

// ============================================================================
// Constructor / Destructor
// ============================================================================

rgbMapTracker::rgbMapTracker()
{
    // Initialize synchronization primitives (PUBLIC - accessed by other classes)
    mutex_rgb_points_vec = std::make_shared<std::mutex>();
    mutex_frame_index = std::make_shared<std::mutex>();
    
    // Initialize public members
    number_of_new_visited_voxel = 0;
    updated_frame_index = 0;
    minimum_depth_for_projection = 0.3;
    maximum_depth_for_projection = 200.0;
    points_rgb_vec_for_projection = nullptr;
    points_2d_vec_for_projection = nullptr;
    
    // Default parameters
    image_obs_cov = 15.0;
    image_obs_threshold = 15.0;
    minimum_pts = 5;
    grid_size = 32;
    
    // Initialize geometric segmentation (DISABLED by default for backwards compatibility)
    geometric_segmentation_enabled_ = false;
    geometric_label_manager_ = new GeometricLabelManager();
    
    // Configure geometric segmentation parameters
    GeometricSegmentationConfig geo_config;
    geo_config.search_radius = 0.12;
    geo_config.min_neighbors_for_normal = 5;
    geo_config.normal_threshold = 0.85;
    geo_config.curvature_threshold = 0.08;
    geo_config.convexity_threshold = -0.015;
    geo_config.region_grow_radius = 0.08;
    geo_config.min_segment_size = 8;
    geo_config.label_majority_threshold = 0.4;
    geo_config.min_labeled_points = 3;
    
    geometric_label_manager_->setConfig(geo_config);
    
    // Label transfer configuration
    depth_discontinuity_threshold_ = 0.25;
    gradient_threshold_ = 30.0;
    
    // Refinement scheduling
    refinement_frame_interval_ = 10;
    refinement_counter_ = 0;
    full_resegmentation_interval_ = 50;
    full_resegmentation_counter_ = 0;
    
    // Statistics
    render_point_count_ = 0;
    frame_index_ = 0;
    total_render_time_ms_ = 0.0;
    total_refinement_time_ms_ = 0.0;
    render_call_count_ = 0;
}

rgbMapTracker::~rgbMapTracker()
{
    if (geometric_label_manager_) {
        delete geometric_label_manager_;
        geometric_label_manager_ = nullptr;
    }
}

// ============================================================================
// Configuration
// ============================================================================

void rgbMapTracker::setGeometricSegmentationConfig(const GeometricSegmentationConfig& config)
{
    if (geometric_label_manager_) {
        geometric_label_manager_->setConfig(config);
    }
}

// ============================================================================
// EXISTING METHODS (from original SR-LIVO - implement these based on your original code)
// ============================================================================

void rgbMapTracker::selectPointsForProjection(
    voxelHashMap& map,
    cloudFrame* p_frame,
    std::vector<rgbPoint*>* rgb_points_vec,
    std::vector<cv::Point2f>* points_2d_vec,
    int window_size,
    int skip
) {
    // Store for later use
    points_rgb_vec_for_projection = rgb_points_vec;
    points_2d_vec_for_projection = points_2d_vec;
    
    rgb_points_vec->clear();
    points_2d_vec->clear();
    
    double u, v;
    Eigen::Vector3d point_world;
    
    for (auto it = map.begin(); it != map.end(); ++it) {
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        for (int i = 0; i < voxel_block.NumPoints(); i += skip) {
            rgbPoint& point = voxel_block.points[i];
            point_world = point.getPosition();
            
            // Check depth
            Eigen::Vector3d point_camera = p_frame->p_state->R_world_camera.transpose() * 
                                           (point_world - p_frame->p_state->t_world_camera);
            
            if (point_camera.z() < minimum_depth_for_projection || 
                point_camera.z() > maximum_depth_for_projection) {
                continue;
            }
            
            // Project to image
            if (p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0)) {
                // Check within window
                int u_i = static_cast<int>(u);
                int v_i = static_cast<int>(v);
                
                if (u_i >= window_size && u_i < p_frame->rgb_image.cols - window_size &&
                    v_i >= window_size && v_i < p_frame->rgb_image.rows - window_size) {
                    rgb_points_vec->push_back(&point);
                    points_2d_vec->push_back(cv::Point2f(u, v));
                }
            }
        }
    }
}

void rgbMapTracker::updatePoseForProjection(cloudFrame* p_frame, double fov_margin)
{
    // Update projection for all stored points
    if (points_rgb_vec_for_projection == nullptr || 
        points_2d_vec_for_projection == nullptr) {
        return;
    }
    
    points_2d_vec_for_projection->clear();
    
    double u, v;
    std::vector<rgbPoint*> valid_points;
    
    for (rgbPoint* point : *points_rgb_vec_for_projection) {
        if (p_frame->project3dPointInThisImage(point->getPosition(), u, v, nullptr, 1.0)) {
            int u_i = static_cast<int>(u);
            int v_i = static_cast<int>(v);
            
            int margin = static_cast<int>(fov_margin * p_frame->rgb_image.cols);
            
            if (u_i >= margin && u_i < p_frame->rgb_image.cols - margin &&
                v_i >= margin && v_i < p_frame->rgb_image.rows - margin) {
                valid_points.push_back(point);
                points_2d_vec_for_projection->push_back(cv::Point2f(u, v));
            }
        }
    }
    
    *points_rgb_vec_for_projection = valid_points;
}

void rgbMapTracker::refreshPointsForProjection(voxelHashMap& map)
{
    // Refresh the projection points list
    // This is typically called after map updates
}

// ============================================================================
// Main Rendering Entry Point
// ============================================================================

void rgbMapTracker::renderPointsInRecentVoxel(
    voxelHashMap& map,
    cloudFrame* p_frame,
    std::vector<voxelId>* voxels_for_render,
    const double& obs_time
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Copy voxel IDs for parallel processing
    g_voxel_for_render_.clear();
    g_voxel_for_render_.reserve(voxels_for_render->size());
    
    for (const auto& vid : *voxels_for_render) {
        g_voxel_for_render_.push_back(vid);
    }
    
    int number_of_voxels = g_voxel_for_render_.size();
    render_point_count_ = 0;
    
    if (number_of_voxels == 0) {
        return;
    }
    
    // Check if geometric segmentation is enabled
    if (geometric_segmentation_enabled_) {
        // Prepare grayscale image for edge detection
        cv::Mat gray_image;
        if (!p_frame->rgb_image.empty()) {
            if (p_frame->rgb_image.channels() == 3) {
                cv::cvtColor(p_frame->rgb_image, gray_image, cv::COLOR_BGR2GRAY);
            } else {
                gray_image = p_frame->rgb_image.clone();
            }
        }
        
        // Clear pending points for this frame
        {
            std::lock_guard<std::mutex> lock(mutex_pending_points_);
            frame_pending_points_.clear();
        }
        
        // Parallel rendering with geometric-aware label transfer
        cv::parallel_for_(cv::Range(0, number_of_voxels), [&](const cv::Range& r) {
            threadRenderPointsInVoxelGeometric(
                map, r.start, r.end, p_frame,
                &g_voxel_for_render_, obs_time, gray_image
            );
        });
        
        // Collect pending points for segmentation
        {
            std::lock_guard<std::mutex> lock(mutex_pending_points_);
            for (rgbPoint* p : frame_pending_points_) {
                pending_points_for_segmentation_.push_back(p);
            }
        }
        
        // Increment counters
        frame_index_++;
        refinement_counter_++;
        full_resegmentation_counter_++;
        
        // Periodic refinement
        if (refinement_counter_ >= refinement_frame_interval_) {
            refineSemanticLabels(false);
            refinement_counter_ = 0;
        }
        
        // Periodic full resegmentation
        if (full_resegmentation_counter_ >= full_resegmentation_interval_) {
            refineSemanticLabels(true);
            full_resegmentation_counter_ = 0;
        }
    } else {
        // Original rendering without geometric segmentation
        cv::parallel_for_(cv::Range(0, number_of_voxels), [&](const cv::Range& r) {
            threadRenderPointsInVoxel(
                map, r.start, r.end, p_frame,
                &g_voxel_for_render_, obs_time
            );
        });
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    total_render_time_ms_ += elapsed_ms;
    render_call_count_++;
    
    // Update frame index
    mutex_frame_index->lock();
    updated_frame_index++;
    mutex_frame_index->unlock();
}

// ============================================================================
// Original Rendering Thread (without geometric segmentation)
// ============================================================================

void rgbMapTracker::threadRenderPointsInVoxel(
    voxelHashMap& map,
    const int& voxel_start,
    const int& voxel_end,
    cloudFrame* p_frame,
    const std::vector<voxelId>* voxels_for_render,
    const double obs_time
) {
    Eigen::Vector3d point_world;
    Eigen::Vector3d point_color;
    double u, v;
    double point_camera_norm;
    
    for (int voxel_index = voxel_start; voxel_index < voxel_end; voxel_index++) {
        const voxelId& vid = (*voxels_for_render)[voxel_index];
        
        auto it = map.find(voxel(vid.kx, vid.ky, vid.kz));
        if (it == map.end()) continue;
        
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        
        for (int point_index = 0; point_index < voxel_block.NumPoints(); point_index++) {
            rgbPoint& point = voxel_block.points[point_index];
            point_world = point.getPosition();
            
            if (!p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0)) {
                continue;
            }
            
            point_camera_norm = (point_world - p_frame->p_state->t_world_camera).norm();
            point_color = p_frame->getRgb(u, v, 0);
            
            mutex_rgb_points_vec->lock();
            
            if (point.updateRgb(
                point_color,
                point_camera_norm,
                Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov),
                obs_time
            )) {
                render_point_count_++;
            }
            
            // Original direct label transfer (causes edge bleeding)
            if (!p_frame->semantic_masks.empty()) {
                int u_i = static_cast<int>(std::round(u));
                int v_i = static_cast<int>(std::round(v));
                
                for (size_t m = 0; m < p_frame->semantic_masks.size(); m++) {
                    const cv::Mat& mask = std::get<0>(p_frame->semantic_masks[m]);
                    int label = std::get<1>(p_frame->semantic_masks[m]);
                    int32_t inst_id = std::get<2>(p_frame->semantic_masks[m]);
                    if (u_i >= 0 && u_i < mask.cols && v_i >= 0 && v_i < mask.rows) {
                        if (mask.at<uint8_t>(v_i, u_i) == 255) {
                            point.updateSemanticLabel(label);
                            point.setInstanceId(inst_id);
                            break;
                        }
                    }
                }
            }
            
            mutex_rgb_points_vec->unlock();
        }
    }
}

// ============================================================================
// Geometric-Aware Rendering Thread
// ============================================================================

void rgbMapTracker::threadRenderPointsInVoxelGeometric(
    voxelHashMap& map,
    const int& voxel_start,
    const int& voxel_end,
    cloudFrame* p_frame,
    const std::vector<voxelId>* voxels_for_render,
    const double obs_time,
    const cv::Mat& gray_image
) {
    Eigen::Vector3d point_world;
    Eigen::Vector3d point_color;
    double u, v;
    double point_camera_norm;
    
    std::vector<rgbPoint*> local_pending_points;
    
    for (int voxel_index = voxel_start; voxel_index < voxel_end; voxel_index++) {
        const voxelId& vid = (*voxels_for_render)[voxel_index];
        
        auto it = map.find(voxel(vid.kx, vid.ky, vid.kz));
        if (it == map.end()) continue;
        
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        
        for (int point_index = 0; point_index < voxel_block.NumPoints(); point_index++) {
            rgbPoint& point = voxel_block.points[point_index];
            point_world = point.getPosition();
            
            if (!p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0)) {
                continue;
            }
            
            point_camera_norm = (point_world - p_frame->p_state->t_world_camera).norm();
            
            int u_i = static_cast<int>(std::round(u));
            int v_i = static_cast<int>(std::round(v));
            
            if (u_i < 0 || u_i >= p_frame->rgb_image.cols ||
                v_i < 0 || v_i >= p_frame->rgb_image.rows) {
                continue;
            }
            
            // RGB Update
            point_color = p_frame->getRgb(u, v, 0);
            
            mutex_rgb_points_vec->lock();
            
            if (point.updateRgb(
                point_color,
                point_camera_norm,
                Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov),
                obs_time
            )) {
                render_point_count_++;
            }
            
            // Geometric-Aware Semantic Label Transfer
            if (!p_frame->semantic_masks.empty()) {
                LabelTransferInfo transfer_info = computeGeometricAwareLabelTransfer(
                    point_world, p_frame,
                    u, v, u_i, v_i,
                    point_camera_norm, gray_image
                );
                
                if (transfer_info.label > 0) {
                    // Register point for geometric segmentation if new
                    if (!point.hasSegmentAssignment()) {
                        local_pending_points.push_back(&point);
                    }
                    
                    // Add label vote to geometric label manager
                    if (geometric_label_manager_) {
                        geometric_label_manager_->onLabelProjected(
                            &point,
                            transfer_info.label,
                            transfer_info.instance_id,
                            transfer_info.confidence
                        );
                    }
                    
                    // Also store raw vote for fallback
                    if (!transfer_info.is_boundary && !transfer_info.is_depth_discontinuity) {
                        point.addSemanticVote(transfer_info.label, transfer_info.confidence);
                    } else {
                        point.addSemanticVote(transfer_info.label, transfer_info.confidence * 0.3f);
                    }
                }
            }
            
            mutex_rgb_points_vec->unlock();
        }
    }
    
    // Add local pending points to shared collection
    if (!local_pending_points.empty()) {
        std::lock_guard<std::mutex> lock(mutex_pending_points_);
        for (rgbPoint* p : local_pending_points) {
            frame_pending_points_.push_back(p);
        }
    }
}

// ============================================================================
// Geometric-Aware Label Transfer
// ============================================================================

LabelTransferInfo rgbMapTracker::computeGeometricAwareLabelTransfer(
    const Eigen::Vector3d& point_world,
    cloudFrame* p_frame,
    double u, double v,
    int u_i, int v_i,
    double point_depth,
    const cv::Mat& gray_image
) {
    LabelTransferInfo result;
    result.label = 0;
    result.instance_id = 0;
    result.confidence = 0.0f;
    result.is_boundary = false;
    result.is_depth_discontinuity = false;
    
    // Check image gradient boundary
    if (!gray_image.empty()) {
        result.is_boundary = isAtImageBoundary(gray_image, u_i, v_i);
    }
    
    // Check depth discontinuity
    result.is_depth_discontinuity = checkDepthDiscontinuity(p_frame, u_i, v_i, point_depth);
    
    // Find semantic label from masks
    for (size_t m = 0; m < p_frame->semantic_masks.size(); m++) {
        const cv::Mat& mask = std::get<0>(p_frame->semantic_masks[m]);
        int label = std::get<1>(p_frame->semantic_masks[m]);
        int32_t inst_id = std::get<2>(p_frame->semantic_masks[m]);
        if (u_i >= 0 && u_i < mask.cols && v_i >= 0 && v_i < mask.rows) {
            if (mask.at<uint8_t>(v_i, u_i) == 255) {
                result.label = label;
                result.instance_id = inst_id;
                result.confidence = computeMaskConfidence(mask, u_i, v_i);
                break;
            }
        }
    }
    
    // Adjust confidence based on boundary detection
    if (result.label > 0 && (result.is_boundary || result.is_depth_discontinuity)) {
        result.confidence *= 0.2f;
    }
    
    return result;
}

bool rgbMapTracker::isAtImageBoundary(const cv::Mat& gray_image, int u, int v)
{
    if (u < 1 || u >= gray_image.cols - 1 || v < 1 || v >= gray_image.rows - 1) {
        return true;
    }
    
    // Sobel gradient magnitude
    float gx = -static_cast<float>(gray_image.at<uchar>(v-1, u-1))
             + static_cast<float>(gray_image.at<uchar>(v-1, u+1))
             - 2.0f * static_cast<float>(gray_image.at<uchar>(v, u-1))
             + 2.0f * static_cast<float>(gray_image.at<uchar>(v, u+1))
             - static_cast<float>(gray_image.at<uchar>(v+1, u-1))
             + static_cast<float>(gray_image.at<uchar>(v+1, u+1));
    
    float gy = -static_cast<float>(gray_image.at<uchar>(v-1, u-1))
             - 2.0f * static_cast<float>(gray_image.at<uchar>(v-1, u))
             - static_cast<float>(gray_image.at<uchar>(v-1, u+1))
             + static_cast<float>(gray_image.at<uchar>(v+1, u-1))
             + 2.0f * static_cast<float>(gray_image.at<uchar>(v+1, u))
             + static_cast<float>(gray_image.at<uchar>(v+1, u+1));
    
    float magnitude = std::sqrt(gx * gx + gy * gy);
    return magnitude > gradient_threshold_;
}

bool rgbMapTracker::checkDepthDiscontinuity(
    cloudFrame* p_frame,
    int u, int v,
    double point_depth
) {
    int half_window = 3;
    int width = p_frame->rgb_image.cols;
    int height = p_frame->rgb_image.rows;
    
    for (size_t m = 0; m < p_frame->semantic_masks.size(); m++) {
        const cv::Mat& mask = std::get<0>(p_frame->semantic_masks[m]);
        bool current_in_mask = false;
        if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {
            current_in_mask = (mask.at<uint8_t>(v, u) == 255);
        }
        
        if (current_in_mask) {
            for (int dv = -half_window; dv <= half_window; dv++) {
                for (int du = -half_window; du <= half_window; du++) {
                    if (du == 0 && dv == 0) continue;
                    
                    int nu = u + du;
                    int nv = v + dv;
                    
                    if (nu >= 0 && nu < width && nv >= 0 && nv < height) {
                        bool neighbor_in_mask = (mask.at<uint8_t>(nv, nu) == 255);
                        if (!neighbor_in_mask) {
                            return true;  // At mask boundary
                        }
                    }
                }
            }
        }
    }
    
    return false;
}

float rgbMapTracker::computeMaskConfidence(const cv::Mat& mask, int u, int v)
{
    int radius = 5;
    int inside_count = 0;
    int total_count = 0;
    
    for (int dv = -radius; dv <= radius; dv++) {
        for (int du = -radius; du <= radius; du++) {
            int nu = u + du;
            int nv = v + dv;
            
            if (nu >= 0 && nu < mask.cols && nv >= 0 && nv < mask.rows) {
                total_count++;
                if (mask.at<uint8_t>(nv, nu) == 255) {
                    inside_count++;
                }
            }
        }
    }
    
    if (total_count == 0) return 0.0f;
    
    float ratio = static_cast<float>(inside_count) / total_count;
    return ratio * ratio;  // Non-linear penalty for edge pixels
}

// ============================================================================
// Semantic Label Refinement
// ============================================================================

void rgbMapTracker::refineSemanticLabels(bool full_resegmentation)
{
    if (!geometric_label_manager_ || !geometric_segmentation_enabled_) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Add pending points
    if (!pending_points_for_segmentation_.empty()) {
        geometric_label_manager_->onPointsAdded(pending_points_for_segmentation_);
        pending_points_for_segmentation_.clear();
    }
    
    // Perform segmentation and label propagation
    if (full_resegmentation) {
        geometric_label_manager_->fullUpdate();
    } else {
        geometric_label_manager_->update();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    total_refinement_time_ms_ += elapsed_ms;
}

void rgbMapTracker::applyRefinedLabelsToMap(voxelHashMap& map)
{
    if (!geometric_label_manager_ || !geometric_segmentation_enabled_) {
        return;
    }
    
    int updated_count = 0;
    int total_count = 0;
    
    for (auto it = map.begin(); it != map.end(); ++it) {
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        for (int i = 0; i < voxel_block.NumPoints(); i++) {
            rgbPoint& point = voxel_block.points[i];
            total_count++;
            
            int refined_label, refined_instance;
            if (geometric_label_manager_->getRefinedLabel(&point, refined_label, refined_instance)) {
                if (refined_label > 0) {
                    point.setFinalSemanticLabel(refined_label);
                    point.setFinalInstanceId(refined_instance);
                    updated_count++;
                }
            } else {
                point.finalizeSemanticLabel();
            }
        }
    }
    
    std::cout << "[rgbMapTracker] Applied refined labels: " 
              << updated_count << "/" << total_count << " points" << std::endl;
}

// ============================================================================
// Export
// ============================================================================

void rgbMapTracker::exportSemanticPointCloud(voxelHashMap& map, const std::string& filename)
{
    if (geometric_segmentation_enabled_) {
        applyRefinedLabelsToMap(map);
    }
    
    std::ofstream ply_file(filename);
    if (!ply_file.is_open()) {
        std::cerr << "[rgbMapTracker] Failed to open: " << filename << std::endl;
        return;
    }
    
    // Count points
    int total_points = 0;
    for (auto it = map.begin(); it != map.end(); ++it) {
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        total_points += voxel_block.NumPoints();
    }
    
    // PLY header
    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << total_points << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property uchar red\n";
    ply_file << "property uchar green\n";
    ply_file << "property uchar blue\n";
    ply_file << "property int semantic_label\n";
    ply_file << "property int instance_id\n";
    ply_file << "end_header\n";
    
    // Write points
    for (auto it = map.begin(); it != map.end(); ++it) {
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        for (int i = 0; i < voxel_block.NumPoints(); i++) {
            rgbPoint& point = voxel_block.points[i];
            Eigen::Vector3d pos = point.getPosition();
            Eigen::Vector3d rgb = point.getRgb();
            
            ply_file << pos.x() << " " << pos.y() << " " << pos.z() << " ";
            ply_file << static_cast<int>(rgb.x()) << " ";
            ply_file << static_cast<int>(rgb.y()) << " ";
            ply_file << static_cast<int>(rgb.z()) << " ";
            ply_file << point.getFinalSemanticLabel() << " ";
            ply_file << point.getFinalInstanceId() << "\n";
        }
    }
    
    ply_file.close();
    std::cout << "[rgbMapTracker] Exported: " << filename << " (" << total_points << " points)" << std::endl;
}

// ============================================================================
// Visualization
// ============================================================================

void rgbMapTracker::getSegmentationVisualization(
    voxelHashMap& map,
    std::vector<Eigen::Vector3d>& positions,
    std::vector<Eigen::Vector3i>& colors
) {
    positions.clear();
    colors.clear();
    
    if (!geometric_label_manager_ || !geometric_segmentation_enabled_) {
        return;
    }
    
    auto getSegmentColor = [](int segment_id) -> Eigen::Vector3i {
        double hue = std::fmod(segment_id * 0.618033988749895, 1.0);
        double h = hue * 6.0;
        int i = static_cast<int>(h);
        double f = h - i;
        double q = 1.0 - f;
        double t = f;
        
        double r, g, b;
        switch (i % 6) {
            case 0: r = 1.0; g = t; b = 0.0; break;
            case 1: r = q; g = 1.0; b = 0.0; break;
            case 2: r = 0.0; g = 1.0; b = t; break;
            case 3: r = 0.0; g = q; b = 1.0; break;
            case 4: r = t; g = 0.0; b = 1.0; break;
            case 5: r = 1.0; g = 0.0; b = q; break;
            default: r = g = b = 0.5;
        }
        
        return Eigen::Vector3i(
            static_cast<int>(r * 255),
            static_cast<int>(g * 255),
            static_cast<int>(b * 255)
        );
    };
    
    for (auto it = map.begin(); it != map.end(); ++it) {
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        for (int i = 0; i < voxel_block.NumPoints(); i++) {
            rgbPoint& point = voxel_block.points[i];
            int segment_id = geometric_label_manager_->getSegmentId(
                const_cast<rgbPoint*>(&point)
            );
            
            if (segment_id > 0) {
                positions.push_back(point.getPosition());
                colors.push_back(getSegmentColor(segment_id));
            }
        }
    }
}

void rgbMapTracker::getBoundaryVisualization(
    voxelHashMap& map,
    std::vector<Eigen::Vector3d>& boundary_points
) {
    boundary_points.clear();
    
    if (!geometric_label_manager_ || !geometric_segmentation_enabled_) {
        return;
    }
    
    for (auto it = map.begin(); it != map.end(); ++it) {
        voxelBlock& voxel_block = const_cast<voxelBlock&>(it->second);
        for (int i = 0; i < voxel_block.NumPoints(); i++) {
            rgbPoint& point = voxel_block.points[i];
            if (geometric_label_manager_->isBoundaryPoint(const_cast<rgbPoint*>(&point))) {
                boundary_points.push_back(point.getPosition());
            }
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

void rgbMapTracker::printStatistics() const
{
    std::cout << "\n========== rgbMapTracker Statistics ==========" << std::endl;
    std::cout << "Geometric segmentation: " << (geometric_segmentation_enabled_ ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "Total points: " << rgb_points_vec.size() << std::endl;
    std::cout << "Render calls: " << render_call_count_ << std::endl;
    
    if (render_call_count_ > 0) {
        std::cout << "Avg render time: " << (total_render_time_ms_ / render_call_count_) << " ms" << std::endl;
    }
    
    if (geometric_label_manager_ && geometric_segmentation_enabled_) {
        geometric_label_manager_->printStatistics();
    }
    
    std::cout << "=============================================\n" << std::endl;
}

int rgbMapTracker::getNumSegments() const
{
    if (geometric_label_manager_ && geometric_segmentation_enabled_) {
        return geometric_label_manager_->getNumSegments();
    }
    return 0;
}