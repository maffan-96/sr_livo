/**
 * @file ptv3_semantic_fusion_example.cpp
 * @brief Complete example of PTv3 integration with SR_LIVO
 *
 * This example demonstrates:
 * 1. Loading PTv3 model
 * 2. Extracting features from point clouds
 * 3. Feature-based semantic fusion
 * 4. Hybrid fusion with traditional projection
 *
 * Build:
 *   g++ -std=c++17 ptv3_semantic_fusion_example.cpp \
 *       -I../include \
 *       -L../build \
 *       -lptv3_integration \
 *       -ltorch -ltorch_cpu -lc10 \
 *       -lpcl_common -lpcl_kdtree \
 *       -lopencv_core -lopencv_highgui \
 *       -o ptv3_example
 */

#include <iostream>
#include <memory>
#include <chrono>

#include "ptv3FeatureExtractor.h"
#include "ptv3Integration.h"
#include "cloudMap.h"
#include "lioOptimization.h"

// ============================================================================
// Example 1: Basic PTv3 Feature Extraction
// ============================================================================

void example1_basic_feature_extraction() {
    std::cout << "\n=== Example 1: Basic Feature Extraction ===\n" << std::endl;

    // 1. Configure PTv3
    PTv3Config config;
    config.model_path = "/home/user/models/ptv3/ptv3_semantic.pt";
    config.device = "cuda:0";  // or "cpu"
    config.batch_size = 4096;
    config.feature_dim = 256;
    config.num_classes = 20;
    config.enable_caching = true;

    // 2. Create feature extractor
    PTv3FeatureExtractor extractor(config);

    // 3. Load model
    if (!extractor.loadModel()) {
        std::cerr << "Failed to load PTv3 model!" << std::endl;
        return;
    }

    // 4. Create sample point cloud (in practice, from LiDAR)
    std::vector<point3D> points;
    for (int i = 0; i < 1000; i++) {
        point3D p;
        p.point = Eigen::Vector3d(
            static_cast<double>(rand()) / RAND_MAX * 10.0,
            static_cast<double>(rand()) / RAND_MAX * 10.0,
            static_cast<double>(rand()) / RAND_MAX * 3.0
        );
        points.push_back(p);
    }

    // 5. Extract features
    PTv3Output output;
    auto start = std::chrono::high_resolution_clock::now();

    if (extractor.extractFeatures(points, output)) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count();

        std::cout << "✓ Feature extraction successful!" << std::endl;
        std::cout << "  Points processed: " << output.num_points << std::endl;
        std::cout << "  Inference time: " << duration << " ms" << std::endl;
        std::cout << "  Throughput: " << output.num_points * 1000 / duration << " pts/sec" << std::endl;

        // 6. Examine results
        std::cout << "\nSample results (first 5 points):" << std::endl;
        for (int i = 0; i < std::min(5, output.num_points); i++) {
            std::cout << "  Point " << i << ":" << std::endl;
            std::cout << "    Feature dim: " << output.features[i].size() << std::endl;
            std::cout << "    Semantic label: " << output.semantic_labels[i] << std::endl;
            std::cout << "    Confidence: " << output.semantic_confidences[i] << std::endl;
            std::cout << "    Normal: ["
                      << output.normals[i].x() << ", "
                      << output.normals[i].y() << ", "
                      << output.normals[i].z() << "]" << std::endl;
        }
    } else {
        std::cerr << "✗ Feature extraction failed!" << std::endl;
    }

    // 7. Print statistics
    extractor.printStatistics();
}

// ============================================================================
// Example 2: Feature-Based Semantic Matching
// ============================================================================

void example2_feature_based_matching() {
    std::cout << "\n=== Example 2: Feature-Based Semantic Matching ===\n" << std::endl;

    // Setup (assuming PTv3 is initialized)
    PTv3Config config;
    config.model_path = "/home/user/models/ptv3/ptv3_semantic.pt";
    config.device = "cuda:0";

    auto extractor = std::make_shared<PTv3FeatureExtractor>(config);
    extractor->loadModel();

    // Create semantic fusion system
    PTv3SemanticFusion fusion;
    if (!fusion.initialize(config)) {
        std::cerr << "Failed to initialize PTv3 fusion!" << std::endl;
        return;
    }

    std::cout << "✓ PTv3 semantic fusion initialized" << std::endl;

    // Create sample point cloud and semantic mask
    std::vector<point3D> points;
    cv::Mat semantic_mask = cv::Mat::zeros(480, 640, CV_32S);

    // Fill with sample data...
    // (In practice, points from LiDAR, mask from segmentation network)

    // Extract features
    PTv3Output ptv3_output;
    if (extractor->extractFeatures(points, ptv3_output)) {
        std::cout << "✓ Extracted features for " << ptv3_output.num_points << " points" << std::endl;

        // Compute feature similarities
        std::cout << "\nFeature similarity analysis:" << std::endl;

        for (int i = 0; i < std::min(5, (int)ptv3_output.features.size() - 1); i++) {
            float sim = extractor->computeFeatureSimilarity(
                ptv3_output.features[i],
                ptv3_output.features[i + 1]
            );

            std::cout << "  Similarity between point " << i
                      << " and " << (i + 1) << ": " << sim << std::endl;
        }
    }
}

// ============================================================================
// Example 3: Hybrid Semantic Fusion Pipeline
// ============================================================================

void example3_hybrid_fusion(
    voxelHashMap& map,
    cloudFrame* frame,
    const std::vector<voxelId>& voxels_to_process
) {
    std::cout << "\n=== Example 3: Hybrid Semantic Fusion ===\n" << std::endl;

    // 1. Initialize PTv3 fusion
    PTv3Config ptv3_config;
    ptv3_config.model_path = "/home/user/models/ptv3/ptv3_semantic.pt";
    ptv3_config.device = "cuda:0";
    ptv3_config.batch_size = 4096;

    PTv3SemanticFusion fusion;
    if (!fusion.initialize(ptv3_config)) {
        std::cerr << "PTv3 initialization failed!" << std::endl;
        return;
    }

    // 2. Configure fusion strategy
    PTv3SemanticFusion::FusionConfig fusion_config;

    // Enable all fusion methods
    fusion_config.use_ptv3_direct_prediction = true;
    fusion_config.use_feature_matching = true;
    fusion_config.use_traditional_projection = true;

    // Set priority weights
    fusion_config.ptv3_direct_weight = 0.7f;      // 70% from PTv3
    fusion_config.feature_match_weight = 0.2f;    // 20% from feature matching
    fusion_config.projection_weight = 0.1f;       // 10% from projection

    // Set thresholds
    fusion_config.feature_similarity_threshold = 0.6f;
    fusion_config.search_window_pixels = 20;

    // Enable temporal filtering
    fusion_config.enable_temporal_filtering = true;
    fusion_config.temporal_window_size = 5;
    fusion_config.temporal_consensus_threshold = 0.6f;

    fusion.setFusionConfig(fusion_config);

    std::cout << "✓ Fusion configured:" << std::endl;
    std::cout << "  PTv3 weight: " << fusion_config.ptv3_direct_weight << std::endl;
    std::cout << "  Feature match weight: " << fusion_config.feature_match_weight << std::endl;
    std::cout << "  Projection weight: " << fusion_config.projection_weight << std::endl;

    // 3. Run hybrid fusion
    std::cout << "\nRunning hybrid semantic fusion..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    fusion.hybridSemanticFusion(map, frame, voxels_to_process);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();

    std::cout << "✓ Hybrid fusion completed in " << duration << " ms" << std::endl;

    // 4. Print statistics
    fusion.printStatistics();

    // 5. Analyze results
    int total_labeled = 0;
    std::map<int, int> label_counts;

    for (const auto& voxel_id : voxels_to_process) {
        auto it = map.find(voxel_id);
        if (it != map.end()) {
            for (auto* point : it->second.points) {
                int label = point->getSemanticLabel();
                if (label > 0) {
                    total_labeled++;
                    label_counts[label]++;
                }
            }
        }
    }

    std::cout << "\nSemantic labeling results:" << std::endl;
    std::cout << "  Total labeled points: " << total_labeled << std::endl;
    std::cout << "  Unique labels: " << label_counts.size() << std::endl;

    std::cout << "\nLabel distribution:" << std::endl;
    for (const auto& [label, count] : label_counts) {
        std::cout << "  Label " << label << ": " << count << " points" << std::endl;
    }
}

// ============================================================================
// Example 4: Boundary Refinement with Features
// ============================================================================

void example4_boundary_refinement(voxelHashMap& map) {
    std::cout << "\n=== Example 4: Boundary Refinement ===\n" << std::endl;

    // Initialize PTv3 fusion
    PTv3Config config;
    config.model_path = "/home/user/models/ptv3/ptv3_semantic.pt";
    config.device = "cuda:0";

    PTv3SemanticFusion fusion;
    fusion.initialize(config);

    // Find boundary voxels (voxels at geometric/semantic discontinuities)
    std::vector<voxelId> boundary_voxels;

    for (const auto& [voxel_id, voxel] : map) {
        // Check if voxel is at boundary
        // (In practice, use geometric features or semantic gradients)
        bool at_boundary = false;

        // Simplified boundary check
        if (voxel.points.size() > 5) {
            int label_0 = voxel.points[0]->getSemanticLabel();

            for (size_t i = 1; i < voxel.points.size(); i++) {
                if (voxel.points[i]->getSemanticLabel() != label_0) {
                    at_boundary = true;
                    break;
                }
            }
        }

        if (at_boundary) {
            boundary_voxels.push_back(voxel_id);
        }
    }

    std::cout << "Found " << boundary_voxels.size() << " boundary voxels" << std::endl;

    // Refine boundaries using features
    if (!boundary_voxels.empty()) {
        std::cout << "Refining boundary labels..." << std::endl;

        fusion.refineBoundaryLabelsWithFeatures(map, boundary_voxels);

        std::cout << "✓ Boundary refinement complete" << std::endl;

        // Statistics
        auto stats = fusion.getStatistics();
        std::cout << "  Refined points: " << stats.boundary_points_refined << std::endl;
    }
}

// ============================================================================
// Example 5: Complete Integration with SR_LIVO
// ============================================================================

class PTv3EnabledLIO {
public:
    PTv3EnabledLIO() {
        // Initialize PTv3
        PTv3Config config;
        config.model_path = "/home/user/models/ptv3/ptv3_semantic.pt";
        config.device = "cuda:0";
        config.batch_size = 4096;
        config.enable_caching = true;

        ptv3_fusion_ = std::make_shared<PTv3SemanticFusion>();

        if (ptv3_fusion_->initialize(config)) {
            std::cout << "[LIO] PTv3 initialized successfully!" << std::endl;
            use_ptv3_ = true;
        } else {
            std::cout << "[LIO] PTv3 initialization failed, using traditional fusion" << std::endl;
            use_ptv3_ = false;
        }
    }

    void processFrame(cloudFrame* frame, voxelHashMap& map) {
        // Standard LIO processing
        // ... (existing code)

        // Enhanced semantic fusion with PTv3
        if (use_ptv3_ && frame->semantic_masks.size() > 0) {
            processSemanticWithPTv3(frame, map);
        } else {
            processSemanticTraditional(frame, map);
        }
    }

private:
    void processSemanticWithPTv3(cloudFrame* frame, voxelHashMap& map) {
        std::cout << "[LIO] Processing with PTv3 enhancement..." << std::endl;

        // Get voxels to process
        std::vector<voxelId> voxels_to_process;
        // ... (extract from map)

        // Run hybrid fusion
        ptv3_fusion_->hybridSemanticFusion(map, frame, voxels_to_process);

        // Periodic statistics
        if (frame_count_++ % 100 == 0) {
            ptv3_fusion_->printStatistics();
        }
    }

    void processSemanticTraditional(cloudFrame* frame, voxelHashMap& map) {
        std::cout << "[LIO] Using traditional projection-based fusion..." << std::endl;
        // ... (existing code)
    }

    std::shared_ptr<PTv3SemanticFusion> ptv3_fusion_;
    bool use_ptv3_ = false;
    int frame_count_ = 0;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "==========================================" << std::endl;
    std::cout << "PTv3 Integration Examples for SR_LIVO" << std::endl;
    std::cout << "==========================================" << std::endl;

    try {
        // Run examples
        example1_basic_feature_extraction();

        example2_feature_based_matching();

        // For examples 3-5, you need actual map and frame data
        // Uncomment when integrated into SR_LIVO:

        /*
        voxelHashMap map;
        cloudFrame* frame = nullptr;  // Load from data
        std::vector<voxelId> voxels;

        example3_hybrid_fusion(map, frame, voxels);
        example4_boundary_refinement(map);

        // Full integration
        PTv3EnabledLIO lio;
        lio.processFrame(frame, map);
        */

        std::cout << "\n==========================================" << std::endl;
        std::cout << "Examples completed successfully!" << std::endl;
        std::cout << "==========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
