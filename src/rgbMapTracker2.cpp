#include "rgbMapTracker.h"

rgbMapTracker::rgbMapTracker()
{
	std::vector<point3D> v_point_temp;
	state *p_state_ = new state();
	p_cloud_frame = new cloudFrame(v_point_temp, p_state_);

	minimum_depth_for_projection = 0.1;
    maximum_depth_for_projection = 200;

	recent_visited_voxel_activated_time = 1.0;

	number_of_new_visited_voxel = 0;

	updated_frame_index = -1;

	in_appending_points = false;

	points_rgb_vec_for_projection = nullptr;

    mutex_rgb_points_vec = std::make_shared<std::mutex>();
    mutex_frame_index = std::make_shared<std::mutex>();

    // new changes for supervoxel segmentation
    
    // Initialize segment manager
    segment_manager = new SemanticSegmentManager();
    
    // Configure semantic segmentation
    semantic_config.supervoxel_resolution = 0.3;  // Adjust based on your scene
    semantic_config.depth_discontinuity_threshold = 0.2;
    semantic_config.label_confidence_threshold = 0.5;
    segment_manager->setConfig(semantic_config);

    // new changes for geometric segmentation
    geometric_label_manager = new GeometricLabelManager();
    GeometricSegmentationConfig geo_config;
    geo_config.search_radius = 0.15;
    geo_config.normal_threshold = 0.85;
    geo_config.curvature_threshold = 0.1;
    geo_config.convexity_threshold = -0.02;
    geometric_label_manager->setConfig(geo_config);
}

void rgbMapTracker::refreshPointsForProjection(voxelHashMap &map)
{
	cloudFrame *p_frame = p_cloud_frame;

	if (p_frame->image_cols == 0 || p_frame->image_rows == 0) return;

	if (p_frame->frame_id == updated_frame_index) return;

	std::vector<rgbPoint*> *points_rgb_vec_for_projection_temp = new std::vector<rgbPoint*>();

	selectPointsForProjection(map, p_frame, points_rgb_vec_for_projection_temp, nullptr, 10.0, 1);

	points_rgb_vec_for_projection = points_rgb_vec_for_projection_temp;

    mutex_frame_index->lock();
	updated_frame_index = p_frame->frame_id;
    mutex_frame_index->unlock();
}

void rgbMapTracker::selectPointsForProjection(voxelHashMap &map, cloudFrame *p_frame, std::vector<rgbPoint*> *pc_out_vec, 
	std::vector<cv::Point2f> *pc_2d_out_vec, double minimum_dis, int skip_step, bool use_all_points)
{
	if (pc_out_vec != nullptr)
    {
        pc_out_vec->clear();
    }

    if (pc_2d_out_vec != nullptr)
    {
        pc_2d_out_vec->clear();
    }

    Hash_map_2d<int, int> mask_index;
    Hash_map_2d<int, float> mask_depth;

    std::map<int, cv::Point2f> map_idx_draw_center;
    std::map<int, cv::Point2f> map_idx_draw_center_raw_pose;

    int u, v;
    double u_f, v_f;

    int acc = 0;
    int blk_rej = 0;

    std::vector<rgbPoint*> points_for_projection;

    std::vector<voxelId> boxes_recent_hitted = voxels_recent_visited;

    if ((!use_all_points) && boxes_recent_hitted.size())
    {
    	for(std::vector<voxelId>::iterator it = boxes_recent_hitted.begin(); it != boxes_recent_hitted.end(); it++)
        {
            if (map[voxel((*it).kx, (*it).ky, (*it).kz)].NumPoints() > 0)
            {
                points_for_projection.push_back(&(map[voxel((*it).kx, (*it).ky, (*it).kz)].points.back()));
            }
        }
    }
    else
    {
        mutex_rgb_points_vec->lock();
        points_for_projection = rgb_points_vec;
        mutex_rgb_points_vec->unlock();
    }

    int point_size = points_for_projection.size();

    for (int point_index = 0; point_index < point_size; point_index += skip_step)
    {
        Eigen::Vector3d point_world = points_for_projection[point_index]->getPosition();

        double depth = (point_world - p_frame->p_state->t_world_camera).norm();

        if (depth > maximum_depth_for_projection)
        {
            continue;
        }

        if (depth < minimum_depth_for_projection)
        {
            continue;
        }

        bool res = p_frame->project3dPointInThisImage(point_world, u_f, v_f, nullptr, 1.0);

        if (res == false)
        {
            continue;
        }

        u = std::round(u_f / minimum_dis) * minimum_dis;
        v = std::round(v_f / minimum_dis) * minimum_dis;

        if ((!mask_depth.if_exist(u, v)) || mask_depth.m_map_2d_hash_map[u][v] > depth)
        {
            acc++;

            if (mask_index.if_exist(u, v))
            {
                int old_idx = mask_index.m_map_2d_hash_map[u][v];

                blk_rej++;

                map_idx_draw_center.erase(map_idx_draw_center.find(old_idx));
                map_idx_draw_center_raw_pose.erase(map_idx_draw_center_raw_pose.find(old_idx));
            }

            mask_index.m_map_2d_hash_map[u][v] = (int)point_index;
            mask_depth.m_map_2d_hash_map[u][v] = (float)depth;

            map_idx_draw_center[point_index] = cv::Point2f(v, u);
            map_idx_draw_center_raw_pose[point_index] = cv::Point2f(u_f, v_f);
        }
    }

    if (pc_out_vec != nullptr)
    {
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
            pc_out_vec->push_back(points_for_projection[it->first]);
    }

    if (pc_2d_out_vec != nullptr)
    {
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
            pc_2d_out_vec->push_back(map_idx_draw_center_raw_pose[it->first]);
    }
}

void rgbMapTracker::updatePoseForProjection(cloudFrame *p_frame, double fov_margin)
{
	p_cloud_frame->p_state->fx = p_frame->p_state->fx;
	p_cloud_frame->p_state->fy = p_frame->p_state->fy;
	p_cloud_frame->p_state->cx = p_frame->p_state->cx;
	p_cloud_frame->p_state->cy = p_frame->p_state->cy;

	p_cloud_frame->image_cols = p_frame->image_cols;
	p_cloud_frame->image_rows = p_frame->image_rows;

	p_cloud_frame->p_state->fov_margin = fov_margin;
	p_cloud_frame->frame_id = p_frame->frame_id;

	p_cloud_frame->p_state->q_world_camera = p_frame->p_state->q_world_camera;
	p_cloud_frame->p_state->t_world_camera = p_frame->p_state->t_world_camera;

	p_cloud_frame->rgb_image = p_frame->rgb_image;
	p_cloud_frame->gray_image = p_frame->gray_image;

    p_cloud_frame->refreshPoseForProjection();
}

const double image_obs_cov = 15;
const double process_noise_sigma = 0.1;

std::atomic<long> render_point_count;

void rgbMapTracker::threadRenderPointsInVoxel(voxelHashMap &map, const int &voxel_start, const int &voxel_end, cloudFrame *p_frame, 
	const std::vector<voxelId> *voxels_for_render, const double obs_time)
{
	Eigen::Vector3d point_world;
	Eigen::Vector3d point_color;

	double u, v;
	double point_camera_norm;

	for (int voxel_index = voxel_start; voxel_index < voxel_end; voxel_index++)
	{
        voxelBlock &voxel_block = map[voxel((*voxels_for_render)[voxel_index].kx, (*voxels_for_render)[voxel_index].ky, (*voxels_for_render)[voxel_index].kz)];

        for (int point_index = 0; point_index < voxel_block.NumPoints(); point_index++)
        {
        	auto &point = voxel_block.points[point_index];

        	point_world = point.getPosition();

        	if (p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0) == false) continue;

        	point_camera_norm = (point_world - p_frame->p_state->t_world_camera).norm();

        	point_color = p_frame->getRgb(u, v, 0);

            mutex_rgb_points_vec->lock();
        	if (voxel_block.points[point_index].updateRgb(point_color, point_camera_norm, 
        		Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov), obs_time))
        	{
        		render_point_count++;
        	}


            // Extract and update semantic label (extract)
            if (!p_frame->semantic_masks.empty()) {
                int instance_id = 0;
                int semantic_label = p_frame->getSemantic(u, v, instance_id);
                
                if (semantic_label > 0) {  // Valid semantic label
                    point.updateSemanticLabel(semantic_label);
                    point.setInstanceId(instance_id);  // Store instance ID if needed
                }
            }

            mutex_rgb_points_vec->unlock();
        }
	}
}

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
    
    // Thread-local collection of points needing segmentation
    std::vector<rgbPoint*> local_pending_points;
    
    for (int voxel_index = voxel_start; voxel_index < voxel_end; voxel_index++) {
        const voxelId& vid = (*voxels_for_render)[voxel_index];
        
        auto it = map.find(voxel(vid.kx, vid.ky, vid.kz));
        if (it == map.end()) continue;
        
        voxelBlock& voxel_block = it->second;
        
        for (int point_index = 0; point_index < voxel_block.NumPoints(); point_index++) {
            rgbPoint& point = voxel_block.points[point_index];
            point_world = point.getPosition();
            
            // Project point to image
            if (!p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0)) {
                continue;
            }
            
            // Compute distance from camera
            point_camera_norm = (point_world - p_frame->p_state->t_world_camera).norm();
            
            // Bounds check
            int u_i = static_cast<int>(std::round(u));
            int v_i = static_cast<int>(std::round(v));
            
            if (u_i < 0 || u_i >= p_frame->rgb_image.cols ||
                v_i < 0 || v_i >= p_frame->rgb_image.rows) {
                continue;
            }
            
            // ================================================================
            // RGB Update (existing logic)
            // ================================================================
            point_color = p_frame->getRgb(u, v, 0);
            
            mutex_rgb_points_vec->lock();
            
            if (point.updateRgb(
                point_color, 
                point_camera_norm,
                Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov),
                obs_time
            )) {
                render_point_count++;
            }
            
            // ================================================================
            // Geometric-Aware Semantic Label Transfer
            // ================================================================
            if (!p_frame->semantic_masks.empty()) {
                // Compute label transfer with boundary awareness
                LabelTransferInfo transfer_info = computeGeometricAwareLabelTransfer(
                    point_world,
                    p_frame,
                    u, v, u_i, v_i,
                    point_camera_norm,
                    gray_image
                );
                
                if (transfer_info.label > 0) {
                    // Register point for geometric segmentation if new
                    if (!point.hasSegmentAssignment()) {
                        local_pending_points.push_back(&point);
                    }
                    
                    // Add label vote to geometric label manager
                    // The manager will assign labels at segment level
                    if (geometric_label_manager) {
                        geometric_label_manager->onLabelProjected(
                            &point,
                            transfer_info.label,
                            transfer_info.instance_id,
                            transfer_info.confidence
                        );
                    }
                    
                    // Also store raw vote for fallback
                    // (in case geometric segmentation hasn't processed this point yet)
                    if (!transfer_info.is_boundary && !transfer_info.is_depth_discontinuity) {
                        // High confidence - update directly
                        point.addSemanticVote(transfer_info.label, transfer_info.confidence);
                    } else {
                        // At boundary - low confidence vote
                        point.addSemanticVote(transfer_info.label, transfer_info.confidence * 0.3f);
                    }
                }
            }
            
            mutex_rgb_points_vec->unlock();
        }
    }
    
    // Add local pending points to shared collection
    if (!local_pending_points.empty()) {
        std::lock_guard<std::mutex> lock(mutex_pending_points);
        for (rgbPoint* p : local_pending_points) {
            frame_pending_points.push_back(p);
        }
    }
}


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
    
    int width = p_frame->rgb_image.cols;
    int height = p_frame->rgb_image.rows;
    
    // ========================================================================
    // Step 1: Check image gradient boundary
    // ========================================================================
    if (!gray_image.empty()) {
        result.is_boundary = isAtImageBoundary(gray_image, u_i, v_i);
    }
    
    // ========================================================================
    // Step 2: Check depth discontinuity using mask boundaries
    // ========================================================================
    result.is_depth_discontinuity = checkDepthDiscontinuity(
        p_frame, u_i, v_i, point_depth
    );
    
    // ========================================================================
    // Step 3: Find semantic label from masks
    // ========================================================================
    for (const auto& [mask, label, inst_id] : p_frame->semantic_masks) {
        if (u_i >= 0 && u_i < mask.cols && v_i >= 0 && v_i < mask.rows) {
            if (mask.at<uint8_t>(v_i, u_i) == 255) {
                result.label = label;
                result.instance_id = inst_id;
                
                // Compute confidence based on position within mask
                result.confidence = computeMaskConfidence(mask, u_i, v_i);
                break;
            }
        }
    }
    
    // ========================================================================
    // Step 4: Adjust confidence based on boundary detection
    // ========================================================================
    if (result.label > 0) {
        if (result.is_boundary || result.is_depth_discontinuity) {
            // Significantly reduce confidence at boundaries
            result.confidence *= 0.2f;
        }
    }
    
    return result;
}

bool rgbMapTracker::isAtImageBoundary(
    const cv::Mat& gray_image, 
    int u, int v
) {
    // Need margin for Sobel kernel
    if (u < 1 || u >= gray_image.cols - 1 ||
        v < 1 || v >= gray_image.rows - 1) {
        return true;  // At image edge
    }
    
    // Compute Sobel gradient magnitude using 3x3 kernel
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
    
    return magnitude > gradient_threshold;
}

bool rgbMapTracker::checkDepthDiscontinuity(
    cloudFrame* p_frame,
    int u, int v,
    double point_depth
) {
    // Check if this pixel is near the boundary of any semantic mask
    // A mask boundary often indicates a depth discontinuity
    
    int half_window = 3;  // Check 7x7 neighborhood
    int width = p_frame->rgb_image.cols;
    int height = p_frame->rgb_image.rows;
    
    for (const auto& [mask, label, inst_id] : p_frame->semantic_masks) {
        bool current_in_mask = false;
        if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {
            current_in_mask = (mask.at<uint8_t>(v, u) == 255);
        }
        
        if (current_in_mask) {
            // Check neighborhood for mask boundary
            for (int dv = -half_window; dv <= half_window; dv++) {
                for (int du = -half_window; du <= half_window; du++) {
                    if (du == 0 && dv == 0) continue;
                    
                    int nu = u + du;
                    int nv = v + dv;
                    
                    if (nu >= 0 && nu < width && nv >= 0 && nv < height) {
                        bool neighbor_in_mask = (mask.at<uint8_t>(nv, nu) == 255);
                        
                        if (!neighbor_in_mask) {
                            // At mask boundary - likely depth discontinuity
                            return true;
                        }
                    }
                }
            }
        }
    }
    
    return false;
}

float rgbMapTracker::computeMaskConfidence(
    const cv::Mat& mask, 
    int u, int v
) {
    // Confidence based on distance from mask boundary
    // Points well inside the mask get higher confidence
    
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
    
    // Non-linear mapping: penalize edge pixels more
    // ratio of 1.0 -> confidence 1.0
    // ratio of 0.5 -> confidence ~0.25
    return ratio * ratio;
}

// ============================================================================
// Semantic Label Refinement
// ============================================================================

void rgbMapTracker::refineSemanticLabels(bool full_resegmentation)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!geometric_label_manager) {
        return;
    }
    
    // Add pending points to geometric segmenter
    if (!pending_points_for_segmentation.empty()) {
        geometric_label_manager->onPointsAdded(pending_points_for_segmentation);
        pending_points_for_segmentation.clear();
    }
    
    // Perform geometric segmentation and label propagation
    if (full_resegmentation) {
        geometric_label_manager->fullUpdate();
    } else {
        geometric_label_manager->update();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    total_refinement_time_ms += elapsed_ms;
    
    if (full_resegmentation) {
        std::cout << "[rgbMapTracker] Full resegmentation completed in " 
                  << elapsed_ms << " ms" << std::endl;
    }
}

void rgbMapTracker::applyRefinedLabelsToMap(voxelHashMap& map)
{
    if (!geometric_label_manager) {
        return;
    }
    
    int updated_count = 0;
    int total_count = 0;
    
    for (auto& [key, voxel_block] : map) {
        for (int i = 0; i < voxel_block.NumPoints(); i++) {
            rgbPoint& point = voxel_block.points[i];
            total_count++;
            
            int refined_label, refined_instance;
            if (geometric_label_manager->getRefinedLabel(
                    &point, refined_label, refined_instance)) {
                
                // Only update if we got a valid refined label
                if (refined_label > 0) {
                    point.setFinalSemanticLabel(refined_label);
                    point.setFinalInstanceId(refined_instance);
                    updated_count++;
                }
            } else {
                // Fallback to per-point voting result
                point.finalizeSemanticLabel();
            }
        }
    }
    
    std::cout << "[rgbMapTracker] Applied refined labels: " 
              << updated_count << "/" << total_count << " points" << std::endl;
}





// New method: segment-aware rendering
void rgbMapTracker::threadRenderPointsInVoxelWithSegments(
    voxelHashMap &map, 
    const int &voxel_start, 
    const int &voxel_end, 
    cloudFrame *p_frame, 
    const std::vector<voxelId> *voxels_for_render, 
    const double obs_time
) {
    Eigen::Vector3d point_world;
    Eigen::Vector3d point_color;
    double u, v;
    double point_camera_norm;

    for (int voxel_index = voxel_start; voxel_index < voxel_end; voxel_index++)
    {
        voxelBlock &voxel_block = map[voxel(
            (*voxels_for_render)[voxel_index].kx, 
            (*voxels_for_render)[voxel_index].ky, 
            (*voxels_for_render)[voxel_index].kz
        )];

        for (int point_index = 0; point_index < voxel_block.NumPoints(); point_index++)
        {
            auto &point = voxel_block.points[point_index];
            point_world = point.getPosition();

            if (p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0) == false) 
                continue;

            point_camera_norm = (point_world - p_frame->p_state->t_world_camera).norm();

            // Update RGB (existing logic)
            point_color = p_frame->getRgb(u, v, 0);

            mutex_rgb_points_vec->lock();
            
            if (voxel_block.points[point_index].updateRgb(
                point_color, point_camera_norm, 
                Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov), 
                obs_time
            )) {
                render_point_count++;
            }

            // Assign point to segment if not already
            if (point.getSegmentId() == 0) {
                segment_manager->assignPointToSegment(&point, point_world);
            }

            // Depth-aware semantic label transfer
            if (!p_frame->semantic_masks.empty()) {
                LabelTransferResult transfer_result = segment_manager->computeDepthAwareLabelTransfer(
                    point_world,
                    cv::Mat(),  // depth_image - not available, using mask-based detection
                    cv::Mat(),  // semantic_masks combined - not used directly
                    p_frame->gray_image,
                    u, v,
                    point_camera_norm,
                    p_frame->semantic_masks
                );
                
                if (transfer_result.semantic_label > 0) {
                    // Get the segment this point belongs to
                    SemanticSegment* segment = segment_manager->getSegment(point.getSegmentId());
                    
                    if (segment) {
                        // Add weighted vote to segment (not directly to point)
                        segment->addLabelVote(
                            transfer_result.semantic_label, 
                            transfer_result.confidence
                        );
                        
                        // Store instance ID
                        if (segment->instance_id == 0) {
                            segment->instance_id = transfer_result.instance_id;
                        }
                    }
                    
                    // Also update point directly for backwards compatibility
                    // But with reduced weight if at boundary
                    if (!transfer_result.is_at_boundary && !transfer_result.is_depth_discontinuity) {
                        point.updateSemanticLabel(transfer_result.semantic_label);
                        point.setInstanceId(transfer_result.instance_id);
                    }
                }
            }

            mutex_rgb_points_vec->unlock();
        }
    }
}


void rgbMapTracker::refineSegments() {
    if (segment_manager) {
        segment_manager->refinementStep();
    }
}

std::vector<voxelId> g_voxel_for_render;

// Modify renderPointsInRecentVoxel to use new method
void rgbMapTracker::renderPointsInRecentVoxel(
    voxelHashMap &map, 
    cloudFrame *p_frame, 
    std::vector<voxelId> *voxels_for_render, 
    const double &obs_time
) {
    g_voxel_for_render.clear();
    std::vector<voxelId>().swap(g_voxel_for_render);

    for (std::vector<voxelId>::iterator it = (*voxels_for_render).begin(); 
         it != (*voxels_for_render).end(); it++) {
        g_voxel_for_render.push_back(*it);
    }

    int number_of_voxels = g_voxel_for_render.size();
    render_point_count = 0;

    // Use the new segment-aware rendering
    cv::parallel_for_(cv::Range(0, number_of_voxels), [&](const cv::Range &r) {
        threadRenderPointsInVoxelWithSegments(
            map, r.start, r.end, p_frame, 
            &g_voxel_for_render, obs_time
        );
    });
    
    // Periodic segment refinement (every N frames)
    static int refinement_counter = 0;
    if (++refinement_counter >= 10) {  // Refine every 10 frames
        refineSegments();
        refinement_counter = 0;
    }
}

