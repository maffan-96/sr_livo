#include "semanticSegment.h"
#include "cloudMap.h"
#include <algorithm>
#include <numeric>

// ============================================================================
// SemanticSegment Implementation
// ============================================================================

SemanticSegment::SemanticSegment(int id) 
    : segment_id(id)
    , centroid(Eigen::Vector3d::Zero())
    , normal(Eigen::Vector3d::UnitZ())
    , covariance(Eigen::Matrix3d::Identity())
    , planarity(0.0)
    , semantic_label(0)
    , instance_id(0)
    , total_observations(0)
    , label_confidence(0.0f)
    , geometry_dirty(true)
    , label_dirty(true)
{
}

void SemanticSegment::addPoint(rgbPoint* point) {
    if (point_set.find(point) == point_set.end()) {
        points.push_back(point);
        point_set.insert(point);
        geometry_dirty = true;
    }
}

void SemanticSegment::removePoint(rgbPoint* point) {
    auto it = point_set.find(point);
    if (it != point_set.end()) {
        point_set.erase(it);
        points.erase(std::remove(points.begin(), points.end(), point), points.end());
        geometry_dirty = true;
    }
}

void SemanticSegment::updateGeometry() {
    if (!geometry_dirty || points.empty()) return;
    
    // Compute centroid
    centroid = Eigen::Vector3d::Zero();
    for (const auto& p : points) {
        centroid += p->getPosition();
    }
    centroid /= static_cast<double>(points.size());
    
    // Compute covariance matrix
    covariance = Eigen::Matrix3d::Zero();
    for (const auto& p : points) {
        Eigen::Vector3d diff = p->getPosition() - centroid;
        covariance += diff * diff.transpose();
    }
    covariance /= static_cast<double>(points.size());
    
    // Eigendecomposition for normal and planarity
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    
    // Normal is eigenvector corresponding to smallest eigenvalue
    normal = solver.eigenvectors().col(0);
    
    // Planarity: ratio of smallest to middle eigenvalue
    // High planarity = small ratio (points lie in a plane)
    double sigma1 = std::sqrt(std::abs(eigenvalues(0)));
    double sigma2 = std::sqrt(std::abs(eigenvalues(1)));
    double sigma3 = std::sqrt(std::abs(eigenvalues(2)));
    
    if (sigma3 > 1e-6) {
        planarity = (sigma2 - sigma1) / sigma3;
    } else {
        planarity = 0.0;
    }
    
    geometry_dirty = false;
}

void SemanticSegment::addLabelVote(int label, float weight) {
    if (label <= 0) return;  // Ignore background
    
    label_histogram[label] += weight;
    total_observations++;
    label_dirty = true;
}

void SemanticSegment::updateSemanticLabel() {
    if (!label_dirty || label_histogram.empty()) return;
    
    // Find label with highest weighted vote
    float max_votes = 0;
    float total_votes = 0;
    int best_label = 0;
    
    for (const auto& [label, votes] : label_histogram) {
        total_votes += votes;
        if (votes > max_votes) {
            max_votes = votes;
            best_label = label;
        }
    }
    
    semantic_label = best_label;
    label_confidence = (total_votes > 0) ? (max_votes / total_votes) : 0.0f;
    
    label_dirty = false;
}

bool SemanticSegment::shouldMergeWith(const SemanticSegment& other, 
                                       const SemanticSegmentConfig& config) const {
    // Check spatial proximity
    double distance = (centroid - other.centroid).norm();
    if (distance > config.spatial_proximity_threshold) {
        return false;
    }
    
    // Check normal similarity (both should be reasonably planar)
    if (planarity > config.planarity_threshold && other.planarity > config.planarity_threshold) {
        double normal_sim = std::abs(normal.dot(other.normal));
        if (normal_sim < config.normal_similarity_threshold) {
            return false;
        }
    }
    
    // Check semantic label consistency
    if (semantic_label != 0 && other.semantic_label != 0 && 
        semantic_label != other.semantic_label) {
        // Both have labels but they differ - don't merge
        // Unless one has very low confidence
        if (label_confidence > 0.5 && other.label_confidence > 0.5) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// SemanticSegmentManager Implementation
// ============================================================================

SemanticSegmentManager::SemanticSegmentManager() 
    : next_segment_id_(1)
{
    mutex_segments = std::make_shared<std::mutex>();
}

SemanticSegmentManager::~SemanticSegmentManager() {
    for (auto& [id, segment] : segments_) {
        delete segment;
    }
}

SemanticSegment* SemanticSegmentManager::getOrCreateSegment(const Eigen::Vector3d& position) {
    SuperVoxelKey key(position, config_.supervoxel_resolution);
    
    auto it = supervoxel_to_segment_.find(key);
    if (it != supervoxel_to_segment_.end()) {
        return segments_[it->second];
    }
    
    // Create new segment
    int new_id = next_segment_id_++;
    SemanticSegment* segment = new SemanticSegment(new_id);
    segments_[new_id] = segment;
    supervoxel_to_segment_[key] = new_id;
    
    // Find and record neighbors
    for (const auto& neighbor_key : getNeighborKeys(key)) {
        auto neighbor_it = supervoxel_to_segment_.find(neighbor_key);
        if (neighbor_it != supervoxel_to_segment_.end()) {
            segment->neighbor_segment_ids.insert(neighbor_it->second);
            segments_[neighbor_it->second]->neighbor_segment_ids.insert(new_id);
        }
    }
    
    return segment;
}

SemanticSegment* SemanticSegmentManager::getSegment(int segment_id) {
    auto it = segments_.find(segment_id);
    return (it != segments_.end()) ? it->second : nullptr;
}

void SemanticSegmentManager::assignPointToSegment(rgbPoint* point, const Eigen::Vector3d& position) {
    std::lock_guard<std::mutex> lock(*mutex_segments);
    
    SemanticSegment* segment = getOrCreateSegment(position);
    segment->addPoint(point);
}

std::vector<SuperVoxelKey> SemanticSegmentManager::getNeighborKeys(const SuperVoxelKey& key) {
    std::vector<SuperVoxelKey> neighbors;
    neighbors.reserve(26);
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                neighbors.emplace_back(key.x + dx, key.y + dy, key.z + dz);
            }
        }
    }
    return neighbors;
}

LabelTransferResult SemanticSegmentManager::computeDepthAwareLabelTransfer(
    const Eigen::Vector3d& point_world,
    const cv::Mat& depth_image,
    const cv::Mat& semantic_masks,
    const cv::Mat& rgb_image,
    double u, double v,
    double point_depth,
    const std::vector<std::tuple<cv::Mat, int, int32_t>>& masks
) {
    LabelTransferResult result;
    result.semantic_label = 0;
    result.instance_id = 0;
    result.confidence = 0.0f;
    result.is_at_boundary = false;
    result.is_depth_discontinuity = false;
    
    int u_i = static_cast<int>(std::round(u));
    int v_i = static_cast<int>(std::round(v));
    
    int width = rgb_image.cols;
    int height = rgb_image.rows;
    
    if (u_i < 0 || u_i >= width || v_i < 0 || v_i >= height) {
        return result;
    }
    
    // 1. Check image boundary (gradient-based)
    if (!rgb_image.empty()) {
        result.is_at_boundary = isAtImageBoundary(rgb_image, u, v);
    }
    
    // 2. Check depth discontinuity
    // Sample depth at neighboring pixels and check for large jumps
    int half_win = config_.depth_check_window_size / 2;
    std::vector<double> neighbor_depths;
    
    for (int dv = -half_win; dv <= half_win; dv++) {
        for (int du = -half_win; du <= half_win; du++) {
            if (du == 0 && dv == 0) continue;
            
            int nu = u_i + du;
            int nv = v_i + dv;
            
            if (nu >= 0 && nu < width && nv >= 0 && nv < height) {
                // For each mask, check if this pixel belongs to same or different object
                for (const auto& [mask, label, inst_id] : masks) {
                    if (mask.at<uint8_t>(v_i, u_i) == 255 && 
                        mask.at<uint8_t>(nv, nu) == 255) {
                        // Same mask - expect similar depth
                        // We don't have explicit depth image, so we use this as a proxy
                    } else if (mask.at<uint8_t>(v_i, u_i) == 255 || 
                               mask.at<uint8_t>(nv, nu) == 255) {
                        // At mask boundary - likely depth discontinuity
                        result.is_depth_discontinuity = true;
                    }
                }
            }
        }
    }
    
    // 3. Get semantic label with confidence
    // Only transfer if not at problematic boundary
    if (!result.is_depth_discontinuity || !result.is_at_boundary) {
        // Check each mask
        for (const auto& [mask, label, inst_id] : masks) {
            if (u_i >= 0 && u_i < mask.cols && v_i >= 0 && v_i < mask.rows) {
                if (mask.at<uint8_t>(v_i, u_i) == 255) {
                    result.semantic_label = label;
                    result.instance_id = inst_id;
                    
                    // Compute confidence based on distance from mask boundary
                    // Simple approach: check if we're well inside the mask
                    int inside_count = 0;
                    int check_radius = 3;
                    int total_checks = 0;
                    
                    for (int dv = -check_radius; dv <= check_radius; dv++) {
                        for (int du = -check_radius; du <= check_radius; du++) {
                            int nu = u_i + du;
                            int nv = v_i + dv;
                            if (nu >= 0 && nu < mask.cols && nv >= 0 && nv < mask.rows) {
                                total_checks++;
                                if (mask.at<uint8_t>(nv, nu) == 255) {
                                    inside_count++;
                                }
                            }
                        }
                    }
                    
                    result.confidence = static_cast<float>(inside_count) / total_checks;
                    break;
                }
            }
        }
    } else {
        // At boundary - reduce confidence significantly
        for (const auto& [mask, label, inst_id] : masks) {
            if (u_i >= 0 && u_i < mask.cols && v_i >= 0 && v_i < mask.rows) {
                if (mask.at<uint8_t>(v_i, u_i) == 255) {
                    result.semantic_label = label;
                    result.instance_id = inst_id;
                    result.confidence = 0.2f;  // Low confidence at boundaries
                    break;
                }
            }
        }
    }
    
    return result;
}

bool SemanticSegmentManager::isAtImageBoundary(const cv::Mat& gray_image, double u, double v) {
    int u_i = static_cast<int>(std::round(u));
    int v_i = static_cast<int>(std::round(v));
    
    // Need 3x3 neighborhood for Sobel
    if (u_i < 1 || u_i >= gray_image.cols - 1 || 
        v_i < 1 || v_i >= gray_image.rows - 1) {
        return true;  // At image edge
    }
    
    // Compute Sobel gradient magnitude at this point
    // Using 3x3 Sobel kernels
    float gx = -gray_image.at<uchar>(v_i-1, u_i-1) + gray_image.at<uchar>(v_i-1, u_i+1)
              -2*gray_image.at<uchar>(v_i, u_i-1) + 2*gray_image.at<uchar>(v_i, u_i+1)
              -gray_image.at<uchar>(v_i+1, u_i-1) + gray_image.at<uchar>(v_i+1, u_i+1);
    
    float gy = -gray_image.at<uchar>(v_i-1, u_i-1) - 2*gray_image.at<uchar>(v_i-1, u_i) - gray_image.at<uchar>(v_i-1, u_i+1)
              +gray_image.at<uchar>(v_i+1, u_i-1) + 2*gray_image.at<uchar>(v_i+1, u_i) + gray_image.at<uchar>(v_i+1, u_i+1);
    
    float magnitude = std::sqrt(gx*gx + gy*gy);
    
    return magnitude > config_.gradient_threshold;
}

void SemanticSegmentManager::refinementStep() {
    std::lock_guard<std::mutex> lock(*mutex_segments);
    
    // Update geometry for all dirty segments
    for (auto& [id, segment] : segments_) {
        if (segment->geometry_dirty) {
            segment->updateGeometry();
        }
        if (segment->label_dirty) {
            segment->updateSemanticLabel();
        }
    }
    
    // Find merge candidates
    std::vector<std::pair<int, int>> merge_pairs;
    
    for (auto& [id, segment] : segments_) {
        if (segment->isEmpty()) continue;
        
        for (int neighbor_id : segment->neighbor_segment_ids) {
            if (neighbor_id <= id) continue;  // Avoid duplicates
            
            SemanticSegment* neighbor = getSegment(neighbor_id);
            if (neighbor && !neighbor->isEmpty()) {
                if (segment->shouldMergeWith(*neighbor, config_)) {
                    merge_pairs.emplace_back(id, neighbor_id);
                }
            }
        }
    }
    
    // Perform merges
    for (const auto& [id_a, id_b] : merge_pairs) {
        mergeSegments(id_a, id_b);
    }
    
    // Propagate labels to points
    propagateLabelsToPoints();
}

void SemanticSegmentManager::mergeSegments(int seg_id_a, int seg_id_b) {
    SemanticSegment* seg_a = getSegment(seg_id_a);
    SemanticSegment* seg_b = getSegment(seg_id_b);
    
    if (!seg_a || !seg_b) return;
    
    // Move all points from B to A
    for (rgbPoint* point : seg_b->points) {
        seg_a->addPoint(point);
    }
    
    // Merge label histograms
    for (const auto& [label, votes] : seg_b->label_histogram) {
        seg_a->label_histogram[label] += votes;
    }
    seg_a->total_observations += seg_b->total_observations;
    seg_a->label_dirty = true;
    
    // Update neighbor references
    for (int neighbor_id : seg_b->neighbor_segment_ids) {
        if (neighbor_id != seg_id_a) {
            SemanticSegment* neighbor = getSegment(neighbor_id);
            if (neighbor) {
                neighbor->neighbor_segment_ids.erase(seg_id_b);
                neighbor->neighbor_segment_ids.insert(seg_id_a);
                seg_a->neighbor_segment_ids.insert(neighbor_id);
            }
        }
    }
    seg_a->neighbor_segment_ids.erase(seg_id_b);
    
    // Update supervoxel mapping
    for (auto& [key, id] : supervoxel_to_segment_) {
        if (id == seg_id_b) {
            id = seg_id_a;
        }
    }
    
    // Delete segment B
    delete seg_b;
    segments_.erase(seg_id_b);
}

void SemanticSegmentManager::propagateLabelsToPoints() {
    for (auto& [id, segment] : segments_) {
        if (segment->isEmpty()) continue;
        
        segment->updateSemanticLabel();
        
        // Only propagate if we have confident label
        if (segment->label_confidence >= config_.label_confidence_threshold &&
            segment->total_observations >= config_.min_observations_for_stable_label) {
            
            for (rgbPoint* point : segment->points) {
                // Use segment's stable label instead of per-point voting
                // This effectively smooths labels within segments
                if (segment->semantic_label > 0) {
                    point->updateSemanticLabel(segment->semantic_label);
                    point->setInstanceId(segment->instance_id);
                }
            }
        }
    }
}

int SemanticSegmentManager::numActiveSegments() const {
    int count = 0;
    for (const auto& [id, segment] : segments_) {
        if (!segment->isEmpty()) count++;
    }
    return count;
}