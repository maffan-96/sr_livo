/**
 * @file ptv3FeatureExtractor.cpp
 * @brief Implementation of PTv3 feature extractor for SR_LIVO
 */

#include "ptv3FeatureExtractor.h"
#include <chrono>
#include <algorithm>
#include <numeric>

// ============================================================================
// Constructor / Destructor
// ============================================================================

PTv3FeatureExtractor::PTv3FeatureExtractor()
    : device_(torch::kCPU), model_loaded_(false) {
    config_ = PTv3Config();
}

PTv3FeatureExtractor::PTv3FeatureExtractor(const PTv3Config& config)
    : config_(config), model_loaded_(false) {

    // Set device
    if (config_.device == "cuda" || config_.device == "cuda:0") {
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
            std::cout << "[PTv3] Using CUDA device" << std::endl;
        } else {
            device_ = torch::Device(torch::kCPU);
            std::cout << "[PTv3] CUDA not available, using CPU" << std::endl;
        }
    } else {
        device_ = torch::Device(torch::kCPU);
        std::cout << "[PTv3] Using CPU device" << std::endl;
    }
}

PTv3FeatureExtractor::~PTv3FeatureExtractor() {
    feature_cache_.clear();
}

// ============================================================================
// Initialization
// ============================================================================

bool PTv3FeatureExtractor::loadModel() {
    try {
        std::cout << "[PTv3] Loading model from: " << config_.model_path << std::endl;

        // Load TorchScript model
        model_ = torch::jit::load(config_.model_path);
        model_.to(device_);
        model_.eval();  // Set to evaluation mode

        std::cout << "[PTv3] Model loaded successfully!" << std::endl;
        model_loaded_ = true;

        return true;

    } catch (const c10::Error& e) {
        std::cerr << "[PTv3] Error loading model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

// ============================================================================
// Feature Extraction - Main Interface
// ============================================================================

bool PTv3FeatureExtractor::extractFeatures(
    const std::vector<point3D>& points,
    PTv3Output& output
) {
    if (!model_loaded_) {
        std::cerr << "[PTv3] Model not loaded!" << std::endl;
        return false;
    }

    if (points.size() < config_.min_points_for_inference) {
        std::cerr << "[PTv3] Not enough points for inference: " << points.size() << std::endl;
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Check cache
    if (config_.enable_caching) {
        std::string cache_key = getCacheKey(points);
        if (getCachedFeatures(cache_key, output)) {
            stats_.cache_hits++;
            return true;
        }
        stats_.cache_misses++;
    }

    // Convert points to Eigen matrix
    std::vector<Eigen::Vector3d> point_positions;
    point_positions.reserve(points.size());
    for (const auto& pt : points) {
        point_positions.push_back(pt.point);
    }

    // Preprocess point cloud
    torch::Tensor input_tensor = preprocessPointCloud(point_positions);
    if (!input_tensor.defined()) {
        std::cerr << "[PTv3] Preprocessing failed" << std::endl;
        return false;
    }

    // Move to device
    input_tensor = input_tensor.to(device_);

    // Run inference
    torch::NoGradGuard no_grad;  // Disable gradient computation

    try {
        // PTv3 expects: (coords, features, offsets) or (coords, features)
        // For simplicity, we use coordinates as features initially
        torch::Tensor coords = input_tensor;  // (N, 3)
        torch::Tensor features = coords;      // (N, 3) or could be (N, 1) intensity

        // Create input vector
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(coords);
        inputs.push_back(features);

        // Forward pass
        auto output_dict = model_.forward(inputs).toGenericDict();

        // Extract outputs
        torch::Tensor feature_tensor = output_dict.at("feat").toTensor();  // (N, 256)
        torch::Tensor logits_tensor = output_dict.at("logits").toTensor(); // (N, num_classes)

        // Move back to CPU for processing
        feature_tensor = feature_tensor.to(torch::kCPU);
        logits_tensor = logits_tensor.to(torch::kCPU);

        // Post-process outputs
        postprocessOutput(feature_tensor, logits_tensor, output);
        output.num_points = points.size();
        output.success = true;

        // Extract geometric features if enabled
        if (config_.extract_normals) {
            extractGeometricFeatures(point_positions, output);
        }

        // Update statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        output.inference_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time
        ).count();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_inferences++;
            stats_.total_points_processed += points.size();
            stats_.total_inference_time_ms += output.inference_time_ms;
            stats_.avg_inference_time_ms = stats_.total_inference_time_ms / stats_.total_inferences;
        }

        // Cache results
        if (config_.enable_caching) {
            std::string cache_key = getCacheKey(points);
            cacheFeatures(cache_key, output);
        }

        return true;

    } catch (const c10::Error& e) {
        std::cerr << "[PTv3] Inference error: " << e.what() << std::endl;
        output.success = false;
        return false;
    }
}

bool PTv3FeatureExtractor::extractFeatures(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    PTv3Output& output
) {
    // Convert PCL to point3D vector
    std::vector<point3D> points;
    points.reserve(cloud->size());

    for (const auto& pt : cloud->points) {
        point3D p;
        p.point = Eigen::Vector3d(pt.x, pt.y, pt.z);
        points.push_back(p);
    }

    return extractFeatures(points, output);
}

bool PTv3FeatureExtractor::extractFeaturesForVoxel(
    const std::vector<rgbPoint*>& voxel_points,
    PTv3Output& output
) {
    // Convert rgbPoint* to point3D
    std::vector<point3D> points;
    points.reserve(voxel_points.size());

    for (auto* pt : voxel_points) {
        point3D p;
        p.point = pt->getPosition();
        points.push_back(p);
    }

    return extractFeatures(points, output);
}

bool PTv3FeatureExtractor::extractFeaturesBatch(
    const std::vector<std::vector<point3D>>& point_batches,
    std::vector<PTv3Output>& outputs
) {
    outputs.resize(point_batches.size());

    bool all_success = true;
    for (size_t i = 0; i < point_batches.size(); i++) {
        bool success = extractFeatures(point_batches[i], outputs[i]);
        all_success = all_success && success;
    }

    return all_success;
}

// ============================================================================
// Feature Matching and Fusion
// ============================================================================

float PTv3FeatureExtractor::computeFeatureSimilarity(
    const Eigen::VectorXf& feat1,
    const Eigen::VectorXf& feat2
) const {
    if (feat1.size() != feat2.size()) {
        std::cerr << "[PTv3] Feature dimension mismatch!" << std::endl;
        return 0.0f;
    }

    // Cosine similarity: dot(f1, f2) / (||f1|| * ||f2||)
    float dot_product = feat1.dot(feat2);
    float norm1 = feat1.norm();
    float norm2 = feat2.norm();

    if (norm1 < 1e-6 || norm2 < 1e-6) {
        return 0.0f;
    }

    float cosine_sim = dot_product / (norm1 * norm2);

    // Convert from [-1, 1] to [0, 1]
    return (cosine_sim + 1.0f) / 2.0f;
}

std::vector<int> PTv3FeatureExtractor::findFeatureKNN(
    const Eigen::VectorXf& query_feature,
    const std::vector<Eigen::VectorXf>& database_features,
    int k
) const {
    // Simple brute-force search (for production, use FAISS or similar)
    std::vector<std::pair<float, int>> similarities;
    similarities.reserve(database_features.size());

    for (size_t i = 0; i < database_features.size(); i++) {
        float sim = computeFeatureSimilarity(query_feature, database_features[i]);
        similarities.push_back({sim, i});
    }

    // Sort by similarity (descending)
    std::partial_sort(
        similarities.begin(),
        similarities.begin() + std::min(k, (int)similarities.size()),
        similarities.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Extract indices
    std::vector<int> indices;
    for (int i = 0; i < std::min(k, (int)similarities.size()); i++) {
        indices.push_back(similarities[i].second);
    }

    return indices;
}

std::vector<PTv3FeatureExtractor::FeatureMatch>
PTv3FeatureExtractor::matchFeaturesTo2DSemantics(
    const std::vector<Eigen::VectorXf>& point_features,
    const cv::Mat& semantic_mask,
    const std::vector<cv::Point2f>& projections
) {
    std::vector<FeatureMatch> matches;

    if (point_features.size() != projections.size()) {
        std::cerr << "[PTv3] Feature and projection count mismatch!" << std::endl;
        return matches;
    }

    // For each projected point, search for best semantic match in local window
    for (size_t i = 0; i < point_features.size(); i++) {
        cv::Point2f proj = projections[i];

        // Check if projection is within image bounds
        if (proj.x < 0 || proj.x >= semantic_mask.cols ||
            proj.y < 0 || proj.y >= semantic_mask.rows) {
            continue;
        }

        // Search window (±15 pixels)
        int search_radius = 15;
        float best_score = -1.0f;
        cv::Point2i best_pixel;
        int best_label = 0;

        for (int dy = -search_radius; dy <= search_radius; dy++) {
            for (int dx = -search_radius; dx <= search_radius; dx++) {
                int x = static_cast<int>(proj.x) + dx;
                int y = static_cast<int>(proj.y) + dy;

                if (x < 0 || x >= semantic_mask.cols ||
                    y < 0 || y >= semantic_mask.rows) {
                    continue;
                }

                int label = semantic_mask.at<int>(y, x);
                if (label == 0) continue;  // Skip unlabeled

                // Compute compatibility score (simplified)
                // In practice, you'd extract 2D features from semantic mask region
                float distance_penalty = std::sqrt(dx*dx + dy*dy) / search_radius;
                float score = 1.0f - distance_penalty;

                if (score > best_score) {
                    best_score = score;
                    best_pixel = cv::Point2i(x, y);
                    best_label = label;
                }
            }
        }

        if (best_score > 0.3f) {  // Threshold
            FeatureMatch match;
            match.point_idx = i;
            match.pixel = best_pixel;
            match.semantic_label = best_label;
            match.confidence = best_score;
            match.feature_similarity = best_score;  // Simplified

            matches.push_back(match);
        }
    }

    return matches;
}

// ============================================================================
// Internal Methods
// ============================================================================

torch::Tensor PTv3FeatureExtractor::preprocessPointCloud(
    const std::vector<Eigen::Vector3d>& points
) {
    if (points.empty()) {
        return torch::Tensor();
    }

    // Voxelize if enabled
    std::vector<Eigen::Vector3d> processed_points = points;
    if (config_.voxel_size > 0.0f) {
        processed_points = voxelizePointCloud(points, config_.voxel_size);
    }

    // Convert to tensor (N, 3)
    int n = processed_points.size();
    torch::Tensor coords = torch::zeros({n, 3}, torch::kFloat32);

    auto coords_accessor = coords.accessor<float, 2>();
    for (int i = 0; i < n; i++) {
        coords_accessor[i][0] = processed_points[i].x();
        coords_accessor[i][1] = processed_points[i].y();
        coords_accessor[i][2] = processed_points[i].z();
    }

    // Normalize coordinates (optional, depends on PTv3 training)
    // Centering around mean
    torch::Tensor mean = coords.mean(0, true);
    coords = coords - mean;

    return coords;
}

void PTv3FeatureExtractor::postprocessOutput(
    const torch::Tensor& feature_tensor,
    const torch::Tensor& logits_tensor,
    PTv3Output& output
) {
    int n = feature_tensor.size(0);
    int feat_dim = feature_tensor.size(1);
    int num_classes = logits_tensor.size(1);

    output.resize(n);

    auto feat_accessor = feature_tensor.accessor<float, 2>();
    auto logits_accessor = logits_tensor.accessor<float, 2>();

    for (int i = 0; i < n; i++) {
        // Extract feature vector
        output.features[i].resize(feat_dim);
        for (int j = 0; j < feat_dim; j++) {
            output.features[i](j) = feat_accessor[i][j];
        }

        // Extract semantic prediction (argmax)
        int best_class = 0;
        float best_logit = logits_accessor[i][0];

        for (int c = 1; c < num_classes; c++) {
            if (logits_accessor[i][c] > best_logit) {
                best_logit = logits_accessor[i][c];
                best_class = c;
            }
        }

        output.semantic_labels[i] = best_class;

        // Compute confidence (softmax of best class)
        std::vector<float> logits(num_classes);
        float max_logit = best_logit;
        for (int c = 0; c < num_classes; c++) {
            logits[c] = logits_accessor[i][c];
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += std::exp(logits[c] - max_logit);
        }
        float confidence = std::exp(best_logit - max_logit) / sum_exp;

        output.semantic_confidences[i] = confidence;
    }
}

void PTv3FeatureExtractor::extractGeometricFeatures(
    const std::vector<Eigen::Vector3d>& points,
    PTv3Output& output
) {
    // Build KD-tree for normal estimation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(points.size());

    for (size_t i = 0; i < points.size(); i++) {
        (*cloud)[i].x = points[i].x();
        (*cloud)[i].y = points[i].y();
        (*cloud)[i].z = points[i].z();
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    int k_neighbors = 20;

    for (size_t i = 0; i < points.size(); i++) {
        std::vector<int> indices;
        std::vector<float> distances;

        kdtree.nearestKSearch((*cloud)[i], k_neighbors, indices, distances);

        if (indices.size() < 3) {
            output.normals[i] = Eigen::Vector3f::Zero();
            output.curvatures[i] = 0.0f;
            continue;
        }

        // Compute covariance matrix
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int idx : indices) {
            centroid += points[idx];
        }
        centroid /= indices.size();

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int idx : indices) {
            Eigen::Vector3d diff = points[idx] - centroid;
            covariance += diff * diff.transpose();
        }
        covariance /= indices.size();

        // Eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        Eigen::Vector3d eigenvalues = solver.eigenvalues();
        Eigen::Matrix3d eigenvectors = solver.eigenvectors();

        // Normal is smallest eigenvector
        output.normals[i] = eigenvectors.col(0).cast<float>();

        // Curvature is ratio of smallest eigenvalue
        float sum_eigenvalues = eigenvalues.sum();
        output.curvatures[i] = (sum_eigenvalues > 1e-6) ?
                               eigenvalues(0) / sum_eigenvalues : 0.0f;
    }
}

// ============================================================================
// Cache Management
// ============================================================================

std::string PTv3FeatureExtractor::getCacheKey(const std::vector<point3D>& points) {
    // Simple hash based on point positions (for production, use better hash)
    std::hash<double> hasher;
    size_t hash = 0;

    for (size_t i = 0; i < std::min(points.size(), size_t(10)); i++) {
        hash ^= hasher(points[i].point.x());
        hash ^= hasher(points[i].point.y());
        hash ^= hasher(points[i].point.z());
    }

    return std::to_string(hash);
}

bool PTv3FeatureExtractor::getCachedFeatures(
    const std::string& key,
    PTv3Output& output
) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = feature_cache_.find(key);
    if (it != feature_cache_.end()) {
        output = it->second;
        return true;
    }

    return false;
}

void PTv3FeatureExtractor::cacheFeatures(
    const std::string& key,
    const PTv3Output& output
) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Simple cache size management (LRU would be better)
    if (feature_cache_.size() > 100) {
        feature_cache_.clear();  // Clear all when limit reached
    }

    feature_cache_[key] = output;
}

// ============================================================================
// Statistics
// ============================================================================

void PTv3FeatureExtractor::printStatistics() const {
    std::cout << "\n[PTv3] Statistics:" << std::endl;
    std::cout << "  Total inferences: " << stats_.total_inferences << std::endl;
    std::cout << "  Total points: " << stats_.total_points_processed << std::endl;
    std::cout << "  Avg inference time: " << stats_.avg_inference_time_ms << " ms" << std::endl;
    std::cout << "  Cache hits: " << stats_.cache_hits << std::endl;
    std::cout << "  Cache misses: " << stats_.cache_misses << std::endl;

    if (stats_.cache_hits + stats_.cache_misses > 0) {
        float hit_rate = 100.0f * stats_.cache_hits /
                        (stats_.cache_hits + stats_.cache_misses);
        std::cout << "  Cache hit rate: " << hit_rate << "%" << std::endl;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

Eigen::MatrixXd convertPointsToMatrix(const std::vector<point3D>& points) {
    Eigen::MatrixXd matrix(points.size(), 3);

    for (size_t i = 0; i < points.size(); i++) {
        matrix.row(i) = points[i].point.transpose();
    }

    return matrix;
}

Eigen::MatrixXd convertPCLToMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    Eigen::MatrixXd matrix(cloud->size(), 3);

    for (size_t i = 0; i < cloud->size(); i++) {
        matrix(i, 0) = (*cloud)[i].x;
        matrix(i, 1) = (*cloud)[i].y;
        matrix(i, 2) = (*cloud)[i].z;
    }

    return matrix;
}

std::vector<Eigen::Vector3d> voxelizePointCloud(
    const std::vector<Eigen::Vector3d>& points,
    float voxel_size
) {
    // Simple voxel grid filtering
    std::unordered_map<std::string, Eigen::Vector3d> voxel_map;

    for (const auto& pt : points) {
        int vx = static_cast<int>(std::floor(pt.x() / voxel_size));
        int vy = static_cast<int>(std::floor(pt.y() / voxel_size));
        int vz = static_cast<int>(std::floor(pt.z() / voxel_size));

        std::string key = std::to_string(vx) + "_" +
                         std::to_string(vy) + "_" +
                         std::to_string(vz);

        if (voxel_map.find(key) == voxel_map.end()) {
            voxel_map[key] = pt;
        }
    }

    std::vector<Eigen::Vector3d> voxelized_points;
    voxelized_points.reserve(voxel_map.size());

    for (const auto& [key, pt] : voxel_map) {
        voxelized_points.push_back(pt);
    }

    return voxelized_points;
}
