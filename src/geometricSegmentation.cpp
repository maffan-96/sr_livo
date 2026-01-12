/**
 * @file geometricSegmentation.cpp
 * @brief Implementation of Geometric Point Cloud Segmentation
 */

#include "geometricSegmentation.h"
#include "cloudMap.h"  // For rgbPoint
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// GeometricSegment Implementation
// ============================================================================

GeometricSegment::GeometricSegment()
    : segment_id(0)
    , centroid(Eigen::Vector3d::Zero())
    , mean_normal(Eigen::Vector3d::UnitZ())
    , mean_curvature(0.0)
    , planarity(0.0)
    , bbox_min(Eigen::Vector3d::Constant(std::numeric_limits<double>::max()))
    , bbox_max(Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest()))
    , semantic_label(0)
    , instance_id(0)
    , label_confidence(0.0f)
    , total_vote_count(0)
    , geometry_dirty(true)
    , label_dirty(true)
{
}

GeometricSegment::GeometricSegment(int id)
    : segment_id(id)
    , centroid(Eigen::Vector3d::Zero())
    , mean_normal(Eigen::Vector3d::UnitZ())
    , mean_curvature(0.0)
    , planarity(0.0)
    , bbox_min(Eigen::Vector3d::Constant(std::numeric_limits<double>::max()))
    , bbox_max(Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest()))
    , semantic_label(0)
    , instance_id(0)
    , label_confidence(0.0f)
    , total_vote_count(0)
    , geometry_dirty(true)
    , label_dirty(true)
{
}

void GeometricSegment::computeSummary(
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<PointGeometricFeatures>& features
) {
    if (point_indices.empty()) return;
    
    // Compute centroid and bounding box
    centroid = Eigen::Vector3d::Zero();
    bbox_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    bbox_max = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
    
    for (int idx : point_indices) {
        const Eigen::Vector3d& pos = positions[idx];
        centroid += pos;
        bbox_min = bbox_min.cwiseMin(pos);
        bbox_max = bbox_max.cwiseMax(pos);
    }
    centroid /= static_cast<double>(point_indices.size());
    
    // Compute mean normal and curvature
    mean_normal = Eigen::Vector3d::Zero();
    mean_curvature = 0.0;
    int valid_count = 0;
    
    for (int idx : point_indices) {
        if (features[idx].features_valid) {
            mean_normal += features[idx].normal;
            mean_curvature += features[idx].curvature;
            valid_count++;
        }
    }
    
    if (valid_count > 0) {
        double norm = mean_normal.norm();
        if (norm > 1e-6) {
            mean_normal /= norm;
        }
        mean_curvature /= valid_count;
    }
    
    // Compute planarity using PCA
    if (point_indices.size() >= 3) {
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int idx : point_indices) {
            Eigen::Vector3d diff = positions[idx] - centroid;
            covariance += diff * diff.transpose();
        }
        covariance /= static_cast<double>(point_indices.size());
        
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        Eigen::Vector3d eigenvalues = solver.eigenvalues();
        
        double sum_eig = eigenvalues.sum();
        if (sum_eig > 1e-10) {
            // Planarity: (lambda2 - lambda1) / lambda3
            // High value means points lie in a plane
            planarity = (eigenvalues(1) - eigenvalues(0)) / eigenvalues(2);
            planarity = std::max(0.0, std::min(1.0, planarity));
        }
    }
    
    geometry_dirty = false;
}

void GeometricSegment::addLabelVote(int label, float weight, int instance) {
    if (label <= 0 || weight <= 0) return;
    
    label_votes[label] += weight;
    total_vote_count++;
    
    if (instance_id == 0 && instance > 0) {
        instance_id = instance;
    }
    
    label_dirty = true;
}

void GeometricSegment::assignLabel(const GeometricSegmentationConfig& config) {
    if (label_votes.empty()) {
        semantic_label = 0;
        label_confidence = 0.0f;
        label_dirty = false;
        return;
    }
    
    // Find label with highest weighted vote
    float total_weight = 0.0f;
    float max_weight = 0.0f;
    int best_label = 0;
    
    for (auto it = label_votes.begin(); it != label_votes.end(); ++it) {
        int label = it->first;
        float weight = it->second;
        total_weight += weight;
        if (weight > max_weight) {
            max_weight = weight;
            best_label = label;
        }
    }
    
    float fraction = (total_weight > 0) ? (max_weight / total_weight) : 0.0f;
    
    if (fraction >= config.label_majority_threshold && 
        total_vote_count >= config.min_labeled_points) {
        semantic_label = best_label;
        label_confidence = fraction;
    } else {
        semantic_label = 0;
        label_confidence = 0.0f;
    }
    
    label_dirty = false;
}

void GeometricSegment::clearVotes() {
    label_votes.clear();
    total_vote_count = 0;
    semantic_label = 0;
    instance_id = 0;
    label_confidence = 0.0f;
    label_dirty = true;
}

// ============================================================================
// SimpleKDTree Implementation
// ============================================================================

SimpleKDTree::SimpleKDTree() : root_(nullptr), points_(nullptr), is_built_(false) {}

SimpleKDTree::~SimpleKDTree() {
    clear();
}

void SimpleKDTree::clear() {
    if (root_) {
        destroyTree(root_);
        root_ = nullptr;
    }
    points_ = nullptr;
    is_built_ = false;
}

void SimpleKDTree::destroyTree(Node* node) {
    if (node) {
        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }
}

void SimpleKDTree::build(const std::vector<Eigen::Vector3d>& points) {
    clear();
    
    if (points.empty()) return;
    
    points_ = &points;
    
    std::vector<int> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    root_ = buildRecursive(indices, 0);
    is_built_ = true;
}

SimpleKDTree::Node* SimpleKDTree::buildRecursive(std::vector<int>& indices, int depth) {
    if (indices.empty()) return nullptr;
    
    int dim = depth % 3;
    
    // Sort by current dimension
    std::sort(indices.begin(), indices.end(), [this, dim](int a, int b) {
        return (*points_)[a](dim) < (*points_)[b](dim);
    });
    
    int mid = indices.size() / 2;
    
    Node* node = new Node();
    node->point_idx = indices[mid];
    node->split_dim = dim;
    node->split_val = (*points_)[indices[mid]](dim);
    
    std::vector<int> left_indices(indices.begin(), indices.begin() + mid);
    std::vector<int> right_indices(indices.begin() + mid + 1, indices.end());
    
    node->left = buildRecursive(left_indices, depth + 1);
    node->right = buildRecursive(right_indices, depth + 1);
    
    return node;
}

std::vector<int> SimpleKDTree::radiusSearch(
    const Eigen::Vector3d& query, 
    double radius
) const {
    std::vector<int> results;
    
    if (!is_built_ || !root_) return results;
    
    double radius_sq = radius * radius;
    searchRadius(root_, query, radius_sq, results);
    
    return results;
}

void SimpleKDTree::searchRadius(
    Node* node, 
    const Eigen::Vector3d& query, 
    double radius_sq,
    std::vector<int>& results
) const {
    if (!node) return;
    
    // Check this node
    const Eigen::Vector3d& point = (*points_)[node->point_idx];
    double dist_sq = (point - query).squaredNorm();
    
    if (dist_sq <= radius_sq) {
        results.push_back(node->point_idx);
    }
    
    // Check which subtrees to explore
    double diff = query(node->split_dim) - node->split_val;
    
    if (diff < 0) {
        // Query is on left side
        searchRadius(node->left, query, radius_sq, results);
        if (diff * diff <= radius_sq) {
            searchRadius(node->right, query, radius_sq, results);
        }
    } else {
        // Query is on right side
        searchRadius(node->right, query, radius_sq, results);
        if (diff * diff <= radius_sq) {
            searchRadius(node->left, query, radius_sq, results);
        }
    }
}

std::vector<int> SimpleKDTree::knnSearch(
    const Eigen::Vector3d& query,
    int k
) const {
    // Simple implementation using radius search with expanding radius
    // For production, use a proper knn algorithm
    
    if (!is_built_ || !root_ || k <= 0) return {};
    
    double radius = 0.1;
    std::vector<int> results;
    
    while (results.size() < static_cast<size_t>(k) && radius < 100.0) {
        results = radiusSearch(query, radius);
        radius *= 2.0;
    }
    
    // Sort by distance and take k nearest
    std::sort(results.begin(), results.end(), [this, &query](int a, int b) {
        return ((*points_)[a] - query).squaredNorm() < 
               ((*points_)[b] - query).squaredNorm();
    });
    
    if (results.size() > static_cast<size_t>(k)) {
        results.resize(k);
    }
    
    return results;
}

// ============================================================================
// GeometricSegmenter Implementation
// ============================================================================

GeometricSegmenter::GeometricSegmenter()
    : next_segment_id_(1)
    , features_computed_(false)
    , segmentation_done_(false)
    , last_segmented_point_count_(0)
{
    kdtree_ = std::make_unique<SimpleKDTree>();
}

GeometricSegmenter::~GeometricSegmenter() = default;

void GeometricSegmenter::setPoints(const std::vector<Eigen::Vector3d>& positions) {
    points_ = positions;
    
    // Build KD-tree
    kdtree_->build(points_);
    
    // Initialize storage
    features_.resize(points_.size());
    point_label_votes_.resize(points_.size());
    point_instance_ids_.resize(points_.size(), 0);
    point_to_segment_.resize(points_.size(), -1);
    
    features_computed_ = false;
    segmentation_done_ = false;
}

void GeometricSegmenter::computeNormalAndCurvature(
    int point_idx, 
    const std::vector<int>& neighbor_indices
) {
    auto& feat = features_[point_idx];
    
    if (neighbor_indices.size() < static_cast<size_t>(config_.min_neighbors_for_normal)) {
        feat.features_valid = false;
        return;
    }
    
    // Compute centroid of neighborhood
    Eigen::Vector3d centroid = points_[point_idx];
    for (int idx : neighbor_indices) {
        centroid += points_[idx];
    }
    centroid /= (neighbor_indices.size() + 1);
    
    // Compute covariance matrix
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    
    auto addToCov = [&](const Eigen::Vector3d& pt) {
        Eigen::Vector3d diff = pt - centroid;
        covariance += diff * diff.transpose();
    };
    
    addToCov(points_[point_idx]);
    for (int idx : neighbor_indices) {
        addToCov(points_[idx]);
    }
    covariance /= (neighbor_indices.size() + 1);
    
    // Eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    
    // Normal is eigenvector of smallest eigenvalue
    feat.normal = solver.eigenvectors().col(0);
    
    // Consistent normal orientation (heuristic: point away from centroid)
    if (feat.normal.dot(points_[point_idx] - centroid) < 0) {
        feat.normal = -feat.normal;
    }
    
    // Curvature: ratio of smallest eigenvalue to sum
    double sum_eigenvalues = eigenvalues.sum();
    if (sum_eigenvalues > 1e-10) {
        feat.curvature = eigenvalues(0) / sum_eigenvalues;
    } else {
        feat.curvature = 0.0;
    }
    
    feat.features_valid = true;
}

void GeometricSegmenter::computeConvexity(
    int point_idx, 
    const std::vector<int>& neighbor_indices
) {
    auto& feat = features_[point_idx];
    
    if (!feat.features_valid || neighbor_indices.empty()) {
        return;
    }
    
    // Convexity: average signed distance of neighbors from tangent plane
    const Eigen::Vector3d& p = points_[point_idx];
    const Eigen::Vector3d& n = feat.normal;
    
    double sum_signed_dist = 0.0;
    int valid_count = 0;
    
    for (int idx : neighbor_indices) {
        Eigen::Vector3d diff = points_[idx] - p;
        double signed_dist = diff.dot(n);
        sum_signed_dist += signed_dist;
        valid_count++;
    }
    
    if (valid_count > 0) {
        feat.convexity = sum_signed_dist / valid_count;
    } else {
        feat.convexity = 0.0;
    }
}

void GeometricSegmenter::detectBoundaryPoints() {
    for (size_t i = 0; i < features_.size(); i++) {
        auto& feat = features_[i];
        
        if (!feat.features_valid) {
            feat.is_boundary = false;
            continue;
        }
        
        // Check curvature threshold
        if (feat.curvature > config_.curvature_threshold) {
            feat.is_boundary = true;
            continue;
        }
        
        // Check convexity (concave regions are boundaries)
        if (feat.convexity < config_.convexity_threshold) {
            feat.is_boundary = true;
            continue;
        }
        
        // Check for normal discontinuity with neighbors
        auto neighbors = kdtree_->radiusSearch(points_[i], config_.search_radius);
        bool has_discontinuity = false;
        
        for (int neighbor_idx : neighbors) {
            if (neighbor_idx == static_cast<int>(i)) continue;
            if (!features_[neighbor_idx].features_valid) continue;
            
            double normal_dot = std::abs(feat.normal.dot(features_[neighbor_idx].normal));
            if (normal_dot < config_.normal_threshold) {
                has_discontinuity = true;
                break;
            }
        }
        
        feat.is_boundary = has_discontinuity;
    }
}

void GeometricSegmenter::computeFeatures() {
    if (points_.empty()) return;
    
    // Step 1: Compute normals and curvature
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(config_.use_parallel)
    #endif
    for (size_t i = 0; i < points_.size(); i++) {
        auto neighbors = kdtree_->radiusSearch(points_[i], config_.search_radius);
        
        // Remove self from neighbors
        neighbors.erase(
            std::remove(neighbors.begin(), neighbors.end(), static_cast<int>(i)),
            neighbors.end()
        );
        
        // Limit number of neighbors
        if (neighbors.size() > static_cast<size_t>(config_.max_neighbors)) {
            // Keep closest neighbors
            std::partial_sort(
                neighbors.begin(),
                neighbors.begin() + config_.max_neighbors,
                neighbors.end(),
                [this, i](int a, int b) {
                    return (points_[a] - points_[i]).squaredNorm() < 
                           (points_[b] - points_[i]).squaredNorm();
                }
            );
            neighbors.resize(config_.max_neighbors);
        }
        
        computeNormalAndCurvature(i, neighbors);
    }
    
    // Step 2: Compute convexity
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(config_.use_parallel)
    #endif
    for (size_t i = 0; i < points_.size(); i++) {
        auto neighbors = kdtree_->radiusSearch(points_[i], config_.search_radius);
        neighbors.erase(
            std::remove(neighbors.begin(), neighbors.end(), static_cast<int>(i)),
            neighbors.end()
        );
        computeConvexity(i, neighbors);
    }
    
    // Step 3: Detect boundary points
    detectBoundaryPoints();
    
    features_computed_ = true;
}

bool GeometricSegmenter::canGrowTo(int from_idx, int to_idx) const {
    const auto& feat_from = features_[from_idx];
    const auto& feat_to = features_[to_idx];
    
    // Can't grow to invalid points
    if (!feat_from.features_valid || !feat_to.features_valid) {
        return false;
    }
    
    // Can't grow across boundary points
    if (feat_to.is_boundary) {
        return false;
    }
    
    // Check normal similarity
    double normal_sim = std::abs(feat_from.normal.dot(feat_to.normal));
    if (normal_sim < config_.normal_threshold) {
        return false;
    }
    
    // Check for concave junction between points
    Eigen::Vector3d diff = points_[to_idx] - points_[from_idx];
    double dist = diff.norm();
    
    if (dist > 1e-6) {
        diff /= dist;
        
        // If connection direction strongly disagrees with normals,
        // it's likely a crease/edge
        double dot_from = std::abs(diff.dot(feat_from.normal));
        double dot_to = std::abs(diff.dot(feat_to.normal));
        
        if (dot_from > 0.5 && dot_to > 0.5) {
            return false;
        }
    }
    
    return true;
}

void GeometricSegmenter::regionGrow() {
    segments_.clear();
    std::fill(point_to_segment_.begin(), point_to_segment_.end(), -1);
    next_segment_id_ = 1;
    
    // Sort points by curvature (start from flattest regions)
    std::vector<int> sorted_indices(points_.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    
    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [this](int a, int b) {
            const auto& fa = features_[a];
            const auto& fb = features_[b];
            
            // Prefer valid, non-boundary, low-curvature points as seeds
            if (fa.features_valid != fb.features_valid) 
                return fa.features_valid > fb.features_valid;
            if (fa.is_boundary != fb.is_boundary) 
                return fa.is_boundary < fb.is_boundary;
            return fa.curvature < fb.curvature;
        }
    );
    
    // Region growing
    for (int seed_idx : sorted_indices) {
        // Skip if already assigned, invalid, or boundary
        if (point_to_segment_[seed_idx] >= 0) continue;
        if (!features_[seed_idx].features_valid) continue;
        if (features_[seed_idx].is_boundary) continue;
        
        // Start new segment
        int segment_id = next_segment_id_++;
        GeometricSegment segment(segment_id);
        
        std::queue<int> queue;
        queue.push(seed_idx);
        point_to_segment_[seed_idx] = segment_id;
        segment.point_indices.push_back(seed_idx);
        
        while (!queue.empty() && 
               segment.point_indices.size() < static_cast<size_t>(config_.max_segment_size)) {
            int current = queue.front();
            queue.pop();
            
            // Find neighbors
            auto neighbors = kdtree_->radiusSearch(
                points_[current], config_.region_grow_radius
            );
            
            for (int neighbor_idx : neighbors) {
                if (neighbor_idx == current) continue;
                if (point_to_segment_[neighbor_idx] >= 0) continue;
                
                if (canGrowTo(current, neighbor_idx)) {
                    point_to_segment_[neighbor_idx] = segment_id;
                    segment.point_indices.push_back(neighbor_idx);
                    queue.push(neighbor_idx);
                }
            }
        }
        
        // Store segment if large enough
        if (segment.point_indices.size() >= 
            static_cast<size_t>(config_.min_segment_size)) {
            segment.computeSummary(points_, features_);
            segments_[segment_id] = std::move(segment);
        } else {
            // Mark points as unassigned
            for (int idx : segment.point_indices) {
                point_to_segment_[idx] = -1;
            }
            next_segment_id_--;
        }
    }
    
    // Handle unassigned points
    handleUnassignedPoints();
    
    // Update segment neighbors
    updateSegmentNeighbors();
    
    segmentation_done_ = true;
    last_segmented_point_count_ = points_.size();
}

void GeometricSegmenter::handleUnassignedPoints() {
    for (size_t i = 0; i < point_to_segment_.size(); i++) {
        if (point_to_segment_[i] >= 0) continue;
        
        // Find nearest assigned neighbor
        auto neighbors = kdtree_->radiusSearch(points_[i], config_.search_radius * 2);
        
        int best_segment = -1;
        double best_dist_sq = std::numeric_limits<double>::max();
        
        for (int neighbor_idx : neighbors) {
            if (neighbor_idx == static_cast<int>(i)) continue;
            if (point_to_segment_[neighbor_idx] < 0) continue;
            
            double dist_sq = (points_[i] - points_[neighbor_idx]).squaredNorm();
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_segment = point_to_segment_[neighbor_idx];
            }
        }
        
        if (best_segment >= 0 && segments_.count(best_segment)) {
            point_to_segment_[i] = best_segment;
            segments_[best_segment].point_indices.push_back(i);
            segments_[best_segment].geometry_dirty = true;
        } else {
            // Create single-point segment
            int segment_id = next_segment_id_++;
            GeometricSegment segment(segment_id);
            segment.point_indices.push_back(i);
            segment.centroid = points_[i];
            segments_[segment_id] = std::move(segment);
            point_to_segment_[i] = segment_id;
        }
    }
}

void GeometricSegmenter::updateSegmentNeighbors() {
    // Clear existing neighbors
    for (auto it = segments_.begin(); it != segments_.end(); ++it) {
        it->second.neighbor_segment_ids.clear();
    }
    
    // Find neighboring segments based on point proximity
    for (size_t i = 0; i < points_.size(); i++) {
        int seg_id = point_to_segment_[i];
        if (seg_id < 0) continue;
        
        auto neighbors = kdtree_->radiusSearch(points_[i], config_.region_grow_radius * 1.5);
        
        for (int neighbor_idx : neighbors) {
            int neighbor_seg_id = point_to_segment_[neighbor_idx];
            if (neighbor_seg_id >= 0 && neighbor_seg_id != seg_id) {
                if (segments_.count(seg_id) && segments_.count(neighbor_seg_id)) {
                    segments_[seg_id].neighbor_segment_ids.insert(neighbor_seg_id);
                }
            }
        }
    }
}

void GeometricSegmenter::performSegmentation() {
    if (!features_computed_) {
        computeFeatures();
    }
    regionGrow();
}

int GeometricSegmenter::getSegmentId(int point_index) const {
    if (point_index < 0 || point_index >= static_cast<int>(point_to_segment_.size())) {
        return -1;
    }
    return point_to_segment_[point_index];
}

const GeometricSegment* GeometricSegmenter::getSegment(int segment_id) const {
    auto it = segments_.find(segment_id);
    return (it != segments_.end()) ? &(it->second) : nullptr;
}

GeometricSegment* GeometricSegmenter::getSegmentMutable(int segment_id) {
    auto it = segments_.find(segment_id);
    return (it != segments_.end()) ? &(it->second) : nullptr;
}

std::vector<int> GeometricSegmenter::getAllSegmentIds() const {
    std::vector<int> ids;
    ids.reserve(segments_.size());
    for (auto it = segments_.begin(); it != segments_.end(); ++it) {
        ids.push_back(it->first);
    }
    return ids;
}

void GeometricSegmenter::addLabelVoteToPoint(
    int point_index, int label, float weight, int instance
) {
    if (point_index < 0 || point_index >= static_cast<int>(point_label_votes_.size())) {
        return;
    }
    
    if (label > 0) {
        point_label_votes_[point_index][label] += weight;
        if (instance > 0) {
            point_instance_ids_[point_index] = instance;
        }
    }
}

void GeometricSegmenter::propagateLabelsToSegments() {
    // Aggregate point votes to segments
    for (auto seg_it = segments_.begin(); seg_it != segments_.end(); ++seg_it) {
        GeometricSegment& segment = seg_it->second;
        segment.clearVotes();
        
        for (int point_idx : segment.point_indices) {
            for (auto vote_it = point_label_votes_[point_idx].begin(); 
                 vote_it != point_label_votes_[point_idx].end(); ++vote_it) {
                segment.addLabelVote(vote_it->first, vote_it->second, point_instance_ids_[point_idx]);
            }
        }
        
        segment.assignLabel(config_);
    }
}

void GeometricSegmenter::propagateSegmentLabelsToPoints(
    std::vector<int>& out_labels, 
    std::vector<int>& out_instances
) {
    out_labels.resize(points_.size(), 0);
    out_instances.resize(points_.size(), 0);
    
    for (size_t i = 0; i < point_to_segment_.size(); i++) {
        int seg_id = point_to_segment_[i];
        if (seg_id < 0) continue;
        
        auto it = segments_.find(seg_id);
        if (it != segments_.end() && it->second.semantic_label > 0) {
            out_labels[i] = it->second.semantic_label;
            out_instances[i] = it->second.instance_id;
        }
    }
}

void GeometricSegmenter::clearAllLabels() {
    for (auto& votes : point_label_votes_) {
        votes.clear();
    }
    std::fill(point_instance_ids_.begin(), point_instance_ids_.end(), 0);
    
    for (auto it = segments_.begin(); it != segments_.end(); ++it) {
        it->second.clearVotes();
    }
}

int GeometricSegmenter::addPointsIncremental(
    const std::vector<Eigen::Vector3d>& new_positions
) {
    int start_idx = points_.size();
    
    points_.insert(points_.end(), new_positions.begin(), new_positions.end());
    features_.resize(points_.size());
    point_label_votes_.resize(points_.size());
    point_instance_ids_.resize(points_.size(), 0);
    point_to_segment_.resize(points_.size(), -1);
    
    // Rebuild KD-tree
    kdtree_->build(points_);
    
    features_computed_ = false;
    
    return start_idx;
}

void GeometricSegmenter::updateSegmentationIncremental() {
    // For now, just redo full segmentation
    // A more sophisticated approach would only process new points
    process();
}

void GeometricSegmenter::printStatistics() const {
    std::cout << "\n--- GeometricSegmenter Statistics ---" << std::endl;
    std::cout << "Total points: " << points_.size() << std::endl;
    std::cout << "Total segments: " << segments_.size() << std::endl;
    
    if (!segments_.empty()) {
        // Segment size statistics
        std::vector<int> sizes;
        int labeled_segments = 0;
        for (auto it = segments_.begin(); it != segments_.end(); ++it) {
            const GeometricSegment& seg = it->second;
            sizes.push_back(seg.numPoints());
            if (seg.semantic_label > 0) labeled_segments++;
        }
        
        std::sort(sizes.begin(), sizes.end());
        int min_size = sizes.front();
        int max_size = sizes.back();
        int median_size = sizes[sizes.size() / 2];
        
        double mean_size = 0;
        for (int s : sizes) mean_size += s;
        mean_size /= sizes.size();
        
        std::cout << "Segment sizes - Min: " << min_size 
                  << ", Max: " << max_size 
                  << ", Median: " << median_size 
                  << ", Mean: " << std::fixed << std::setprecision(1) << mean_size 
                  << std::endl;
        std::cout << "Labeled segments: " << labeled_segments 
                  << " (" << (100.0 * labeled_segments / segments_.size()) << "%)" 
                  << std::endl;
    }
    
    // Boundary point statistics
    int boundary_count = 0;
    int valid_features = 0;
    for (const auto& feat : features_) {
        if (feat.features_valid) {
            valid_features++;
            if (feat.is_boundary) boundary_count++;
        }
    }
    
    std::cout << "Valid features: " << valid_features 
              << " (" << (100.0 * valid_features / points_.size()) << "%)" 
              << std::endl;
    std::cout << "Boundary points: " << boundary_count 
              << " (" << (100.0 * boundary_count / points_.size()) << "%)" 
              << std::endl;
    std::cout << "-----------------------------------\n" << std::endl;
}

// ============================================================================
// GeometricLabelManager Implementation
// ============================================================================

GeometricLabelManager::GeometricLabelManager()
    : needs_rebuild_(false)
    , cache_valid_(false)
{
}

GeometricLabelManager::~GeometricLabelManager() = default;

void GeometricLabelManager::setConfig(const GeometricSegmentationConfig& config) {
    config_ = config;
    segmenter_.setConfig(config);
}

void GeometricLabelManager::onPointsAdded(const std::vector<rgbPoint*>& points) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (rgbPoint* p : points) {
        if (point_to_index_.find(p) == point_to_index_.end()) {
            pending_points_.push_back(p);
        }
    }
    
    needs_rebuild_ = true;
    cache_valid_ = false;
}

void GeometricLabelManager::onLabelProjected(
    rgbPoint* point, 
    int label, 
    int instance, 
    float confidence
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = point_to_index_.find(point);
    if (it != point_to_index_.end()) {
        // Weight votes by confidence
        float weight = std::max(0.1f, confidence);
        segmenter_.addLabelVoteToPoint(it->second, label, weight, instance);
        cache_valid_ = false;
    }
}

void GeometricLabelManager::update() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Add pending points if any
    if (!pending_points_.empty()) {
        rebuildSegmenter();
    }
    
    // Propagate labels to segments
    segmenter_.propagateLabelsToSegments();
    
    // Update cache
    updateCache();
}

void GeometricLabelManager::fullUpdate() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Force full rebuild
    needs_rebuild_ = true;
    rebuildSegmenter();
    
    // Propagate labels
    segmenter_.propagateLabelsToSegments();
    
    // Update cache
    updateCache();
}

void GeometricLabelManager::rebuildSegmenter() {
    if (pending_points_.empty() && !needs_rebuild_) return;
    
    // Add pending points
    if (!pending_points_.empty()) {
        std::vector<Eigen::Vector3d> all_positions;
        all_positions.reserve(index_to_point_.size() + pending_points_.size());
        
        // Existing points
        for (rgbPoint* p : index_to_point_) {
            all_positions.push_back(p->getPosition());
        }
        
        // New points
        for (rgbPoint* p : pending_points_) {
            int idx = all_positions.size();
            point_to_index_[p] = idx;
            index_to_point_.push_back(p);
            all_positions.push_back(p->getPosition());
        }
        
        pending_points_.clear();
        
        // Rebuild segmenter with all points
        segmenter_.setPoints(all_positions);
    }
    
    // Perform segmentation
    segmenter_.process();
    
    needs_rebuild_ = false;
}

void GeometricLabelManager::updateCache() {
    cached_labels_.clear();
    
    std::vector<int> labels, instances;
    segmenter_.propagateSegmentLabelsToPoints(labels, instances);
    
    for (size_t i = 0; i < index_to_point_.size(); i++) {
        if (labels[i] > 0) {
            cached_labels_[index_to_point_[i]] = std::make_pair(labels[i], instances[i]);
        }
    }
    
    cache_valid_ = true;
}

bool GeometricLabelManager::getRefinedLabel(
    rgbPoint* point, 
    int& out_label, 
    int& out_instance
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!cache_valid_) {
        updateCache();
    }
    
    auto it = cached_labels_.find(point);
    if (it != cached_labels_.end()) {
        out_label = it->second.first;
        out_instance = it->second.second;
        return true;
    }
    
    return false;
}

int GeometricLabelManager::getSegmentId(rgbPoint* point) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = point_to_index_.find(point);
    if (it != point_to_index_.end()) {
        return segmenter_.getSegmentId(it->second);
    }
    return -1;
}

bool GeometricLabelManager::isBoundaryPoint(rgbPoint* point) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = point_to_index_.find(point);
    if (it != point_to_index_.end()) {
        const auto& feat = segmenter_.getFeature(it->second);
        return feat.is_boundary;
    }
    return false;
}

int GeometricLabelManager::getNumSegments() const {
    return segmenter_.numSegments();
}

int GeometricLabelManager::getNumPoints() const {
    return static_cast<int>(index_to_point_.size());
}

void GeometricLabelManager::printStatistics() const {
    std::cout << "\n--- GeometricLabelManager Statistics ---" << std::endl;
    std::cout << "Registered points: " << index_to_point_.size() << std::endl;
    std::cout << "Pending points: " << pending_points_.size() << std::endl;
    std::cout << "Cached labels: " << cached_labels_.size() << std::endl;
    std::cout << "Cache valid: " << (cache_valid_ ? "yes" : "no") << std::endl;
    
    segmenter_.printStatistics();
}

void GeometricLabelManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    point_to_index_.clear();
    index_to_point_.clear();
    pending_points_.clear();
    cached_labels_.clear();
    
    // Reset segmenter by setting empty points
    segmenter_.setPoints({});
    
    needs_rebuild_ = false;
    cache_valid_ = false;
}