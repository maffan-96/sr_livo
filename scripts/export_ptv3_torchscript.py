#!/usr/bin/env python3
"""
Export PTv3 model to TorchScript for C++ integration

This script loads a trained PTv3 model and exports it to TorchScript format
that can be loaded by LibTorch in the SR_LIVO C++ pipeline.

Usage:
    python export_ptv3_torchscript.py --config <path_to_ptv3_config> \
                                      --checkpoint <path_to_weights> \
                                      --output <output_path.pt>
"""

import argparse
import sys
import os
import torch
import torch.nn as nn

# Add PTv3 to path (modify based on your PTv3 installation)
sys.path.append('/home/user/Point-Transformer-V3')

try:
    from pointcept.models import build_model
    from pointcept.datasets import build_dataset
    from pointcept.utils.config import Config
except ImportError:
    print("Error: PTv3 (pointcept) not found. Please install from:")
    print("https://github.com/Pointcept/PointTransformerV3")
    sys.exit(1)


class PTv3Wrapper(nn.Module):
    """
    Wrapper around PTv3 model for TorchScript export

    Provides a simplified interface:
    - Input: coordinates (N, 3), features (N, C)
    - Output: dict with 'feat' (N, 256) and 'logits' (N, num_classes)
    """

    def __init__(self, model, feature_dim=256):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim

    def forward(self, coords, features):
        """
        Forward pass for TorchScript

        Args:
            coords: (N, 3) point coordinates
            features: (N, C) point features (can be same as coords)

        Returns:
            dict with:
                - 'feat': (N, feature_dim) learned features
                - 'logits': (N, num_classes) semantic predictions
        """
        # Prepare input dict (PTv3 format)
        batch_size = 1
        offset = torch.tensor([0, coords.shape[0]], dtype=torch.long)

        input_dict = {
            'coord': coords.float(),
            'feat': features.float(),
            'offset': offset,
            'grid_size': 0.05  # Adjust based on your voxel size
        }

        # Forward pass
        output = self.model(input_dict)

        # Extract features and logits
        # Note: Exact keys depend on PTv3 model architecture
        # Common variants: 'seg_logits', 'feat', 'backbone_feat'
        result = {
            'feat': output.get('feat', output.get('backbone_feat')),
            'logits': output.get('seg_logits', output.get('logits'))
        }

        return result


def load_ptv3_model(config_path, checkpoint_path):
    """Load trained PTv3 model"""

    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)

    print(f"Building model...")
    model = build_model(cfg.model)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Model loaded successfully!")
    return model


def export_to_torchscript(model, output_path, feature_dim=256):
    """Export model to TorchScript format"""

    print("Wrapping model for TorchScript...")
    wrapped_model = PTv3Wrapper(model, feature_dim)
    wrapped_model.eval()

    # Create example input
    example_coords = torch.randn(1000, 3)
    example_features = example_coords  # Use coords as features

    print("Tracing model with example input...")
    try:
        # Option 1: Tracing (recommended for inference)
        traced_model = torch.jit.trace(
            wrapped_model,
            (example_coords, example_features)
        )

        print(f"Saving TorchScript model to: {output_path}")
        traced_model.save(output_path)

    except Exception as e:
        print(f"Tracing failed: {e}")
        print("Trying scripting instead...")

        # Option 2: Scripting (fallback)
        try:
            scripted_model = torch.jit.script(wrapped_model)
            print(f"Saving TorchScript model to: {output_path}")
            scripted_model.save(output_path)
        except Exception as e2:
            print(f"Scripting also failed: {e2}")
            print("Export failed!")
            return False

    print("Export successful!")
    return True


def verify_exported_model(model_path):
    """Verify the exported model can be loaded and run"""

    print(f"\nVerifying exported model: {model_path}")

    try:
        model = torch.jit.load(model_path)
        model.eval()

        # Test inference
        test_coords = torch.randn(500, 3)
        test_features = test_coords

        with torch.no_grad():
            output = model(test_coords, test_features)

        print("Verification passed!")
        print(f"  Output keys: {output.keys()}")
        if 'feat' in output:
            print(f"  Feature shape: {output['feat'].shape}")
        if 'logits' in output:
            print(f"  Logits shape: {output['logits'].shape}")

        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Export PTv3 model to TorchScript for C++ integration'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to PTv3 config file (.py)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PTv3 checkpoint (.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ptv3_semantic.pt',
        help='Output path for TorchScript model (.pt)'
    )
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=256,
        help='Feature dimension (default: 256)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify exported model after export'
    )

    args = parser.parse_args()

    # Check inputs exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    model = load_ptv3_model(args.config, args.checkpoint)

    # Export to TorchScript
    success = export_to_torchscript(model, args.output, args.feature_dim)

    if not success:
        sys.exit(1)

    # Verify if requested
    if args.verify:
        if not verify_exported_model(args.output):
            sys.exit(1)

    print("\n" + "="*60)
    print("Export completed successfully!")
    print(f"Model saved to: {args.output}")
    print("\nTo use in SR_LIVO:")
    print(f"1. Copy {args.output} to your model directory")
    print(f"2. Update PTv3Config::model_path in your code")
    print(f"3. Initialize PTv3FeatureExtractor and call loadModel()")
    print("="*60)


if __name__ == '__main__':
    main()
