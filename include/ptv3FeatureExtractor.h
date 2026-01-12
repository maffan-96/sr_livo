/**
 * @file ptv3FeatureExtractor.h
 * @brief Point Transformer V3 feature extraction interface for SR_LIVO
 *
 * This module provides integration with PTv3 for learned 3D feature extraction
 * and semantic segmentation of LiDAR point clouds.
 */

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <mutex>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

// LibTorch (PyTorch C++ API)
#include <torch/torch.h>
#include <torch/script.h>

// SR_LIVO types
#include "cloudMap.h"

// ============================================================================
// Configuration
// ============================================================================

struct PTv3Config {
    // Model paths
    std::string model_path = "/home/user/models/ptv3_semantic.pt";
    std::string weights_path = "/home/user/models/ptv3_weights.pth";

    // Model parameters
    int feature_dim = 256;           // PTv3 feature dimension
    int num_classes = 20;            // Number of semantic classes (adjust for your dataset)

    // Inference settings
    int batch_size = 4096;           // Points per batch
    int min_points_for_inference = 500;  // Minimum points to run PTv3
    float voxel_size = 0.05f;        // Voxelization for preprocessing (5cm)

    // Feature extraction modes
    bool extract_features = true;     // Extract 256-dim features
    bool predict_semantics = true;    // Predict semantic labels
    bool extract_normals = true;      // Extract geometric normals

    // Device settings
    std::string device = "cuda:0";    // "cuda:0" or "cpu"
    int num_workers = 2;              // Parallel data loading threads

    // Performance settings
    bool use_amp = true;              // Automatic Mixed Precision (faster)
    bool enable_caching = true;       // Cache features for revisited regions
    int cache_size_mb = 512;          // Feature cache size in MB
};

// ============================================================================
// PTv3 Feature Output
// ============================================================================

struct PTv3Output {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Per-point features
    std::vector<Eigen::VectorXf> features;         // (N, 256) learned features
    std::vector<int> semantic_labels;              // (N,) predicted labels
    std::vector<float> semantic_confidences;       // (N,) prediction confidence [0,1]

    // Per-point geometric features (from backbone)
    std::vector<Eigen::Vector3f> normals;          // (N, 3) estimated normals
    std::vector<float> curvatures;                 // (N,) local curvature

    // Metadata
    int num_points = 0;
    double inference_time_ms = 0.0;
    bool success = false;

    void resize(int n) {
        num_points = n;
        features.resize(n);
        semantic_labels.resize(n, 0);
        semantic_confidences.resize(n, 0.0f);
        normals.resize(n);
        curvatures.resize(n, 0.0f);
    }
};

// ============================================================================
// PTv3 Feature Extractor Class
// ============================================================================

class PTv3FeatureExtractor {
public:
    // ========================================================================
    // Constructor / Destructor
    // ========================================================================

    PTv3FeatureExtractor();
    explicit PTv3FeatureExtractor(const PTv3Config& config);
    ~PTv3FeatureExtractor();

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Load PTv3 TorchScript model
     * @return true if successful
     */
    bool loadModel();

    /**
     * @brief Check if model is loaded and ready
     */
    bool isReady() const { return model_loaded_; }

    // ========================================================================
    // Feature Extraction - Main Interface
    // ========================================================================

    /**
     * @brief Extract PTv3 features from point cloud
     * @param points Input point cloud (world coordinates)
     * @param output PTv3 features and predictions
     * @return true if successful
     */
    bool extractFeatures(
        const std::vector<point3D>& points,
        PTv3Output& output
    );

    /**
     * @brief Extract features from PCL point cloud
     */
    bool extractFeatures(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        PTv3Output& output
    );

    /**
     * @brief Extract features for points in a voxel
     */
    bool extractFeaturesForVoxel(
        const std::vector<rgbPoint*>& voxel_points,
        PTv3Output& output
    );

    /**
     * @brief Batch extraction for multiple frames (more efficient)
     */
    bool extractFeaturesBatch(
        const std::vector<std::vector<point3D>>& point_batches,
        std::vector<PTv3Output>& outputs
    );

    // ========================================================================
    // Feature Matching and Fusion
    // ========================================================================

    /**
     * @brief Compute feature similarity between two points
     * @return Cosine similarity in [0, 1]
     */
    float computeFeatureSimilarity(
        const Eigen::VectorXf& feat1,
        const Eigen::VectorXf& feat2
    ) const;

    /**
     * @brief Find k-nearest neighbors in feature space
     */
    std::vector<int> findFeatureKNN(
        const Eigen::VectorXf& query_feature,
        const std::vector<Eigen::VectorXf>& database_features,
        int k = 5
    ) const;

    /**
     * @brief Match 3D point features to 2D semantic regions
     * Used for feature-based semantic fusion
     */
    struct FeatureMatch {
        int point_idx;
        cv::Point2i pixel;
        int semantic_label;
        float confidence;
        float feature_similarity;
    };

    std::vector<FeatureMatch> matchFeaturesTo2DSemantics(
        const std::vector<Eigen::VectorXf>& point_features,
        const cv::Mat& semantic_mask,
        const std::vector<cv::Point2f>& projections
    );

    // ========================================================================
    // Configuration and Utilities
    // ========================================================================

    void setConfig(const PTv3Config& config) { config_ = config; }
    PTv3Config getConfig() const { return config_; }

    void enableFeatureExtraction(bool enable) { config_.extract_features = enable; }
    void enableSemanticPrediction(bool enable) { config_.predict_semantics = enable; }

    // Statistics
    struct Statistics {
        int total_inferences = 0;
        int total_points_processed = 0;
        double total_inference_time_ms = 0.0;
        double avg_inference_time_ms = 0.0;
        int cache_hits = 0;
        int cache_misses = 0;
    };

    Statistics getStatistics() const { return stats_; }
    void resetStatistics() { stats_ = Statistics(); }
    void printStatistics() const;

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * @brief Preprocess point cloud for PTv3 input
     * Applies voxelization, normalization, etc.
     */
    torch::Tensor preprocessPointCloud(
        const std::vector<Eigen::Vector3d>& points
    );

    /**
     * @brief Run PTv3 model inference
     */
    bool runInference(
        const torch::Tensor& input_coords,
        const torch::Tensor& input_features,
        PTv3Output& output
    );

    /**
     * @brief Post-process PTv3 outputs
     */
    void postprocessOutput(
        const torch::Tensor& feature_tensor,
        const torch::Tensor& logits_tensor,
        PTv3Output& output
    );

    /**
     * @brief Extract normals from intermediate layers (optional)
     */
    void extractGeometricFeatures(
        const std::vector<Eigen::Vector3d>& points,
        PTv3Output& output
    );

    /**
     * @brief Cache management for feature reuse
     */
    std::string getCacheKey(const std::vector<point3D>& points);
    bool getCachedFeatures(const std::string& key, PTv3Output& output);
    void cacheFeatures(const std::string& key, const PTv3Output& output);

    // ========================================================================
    // Member Variables
    // ========================================================================

    PTv3Config config_;

    // PyTorch model
    torch::jit::script::Module model_;
    torch::Device device_;
    bool model_loaded_ = false;

    // Feature cache
    std::unordered_map<std::string, PTv3Output> feature_cache_;
    std::mutex cache_mutex_;

    // Statistics
    Statistics stats_;
    std::mutex stats_mutex_;

    // KD-tree for feature space queries (optional)
    // Note: For high-dimensional features, consider approximate NN (FAISS, etc.)
    std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> feature_kdtree_;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Convert SR_LIVO point3D to Eigen matrix for PTv3
 */
Eigen::MatrixXd convertPointsToMatrix(const std::vector<point3D>& points);

/**
 * @brief Convert PCL point cloud to Eigen matrix
 */
Eigen::MatrixXd convertPCLToMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

/**
 * @brief Voxelize point cloud for PTv3 preprocessing
 */
std::vector<Eigen::Vector3d> voxelizePointCloud(
    const std::vector<Eigen::Vector3d>& points,
    float voxel_size
);
