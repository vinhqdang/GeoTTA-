"""
Minimal example to test GeoTTA with 8GB VRAM.
Start with this to verify everything works.
"""

import torch
import clip
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from geotta.models.geometric_bridge import GeometricBridge
from geotta.models.tta_adapter import TestTimeAdapter
from geotta.utils.memory import print_memory_stats, MemoryProfiler
from geotta.utils.visualization import plot_uncertainty_distribution


def create_dummy_config():
    """Create a minimal configuration for testing."""
    return {
        'model': {
            'clip_model': 'ViT-B/32',
            'bridge_dim': 256,  # Smaller for testing
            'bridge_layers': 1,
            'bridge_heads': 4,
            'dropout': 0.1,
            'temperature': 0.07
        },
        'uncertainty': {
            'geometric_weight': 1.0,
            'angular_weight': 0.5,
            'calibration_bins': 15
        },
        'test_time': {
            'adaptation_steps': 1,
            'adaptation_lr': 0.001,
            'cache_size': 32
        }
    }


def load_sample_image():
    """Load a sample image for testing."""
    # Try to create a dummy image if no real image is available
    try:
        # Try to load a real image
        image_path = "test_image.jpg"  # User should provide this
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            # Create a dummy image
            print("Creating dummy test image...")
            dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(dummy_array)
    except:
        # Fallback to completely synthetic image
        dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_array)
    
    return image


def test_clip_loading():
    """Test basic CLIP loading and inference."""
    print("Testing CLIP model loading...")
    
    with MemoryProfiler("CLIP Loading"):
        model, preprocess = clip.load("ViT-B/32", device="cuda")
        print("✓ CLIP model loaded successfully")
        print_memory_stats()
    
    return model, preprocess


def test_geometric_bridge(config):
    """Test GeometricBridge initialization and forward pass."""
    print("\nTesting GeometricBridge...")
    
    with MemoryProfiler("GeometricBridge"):
        # Initialize model
        bridge = GeometricBridge(config)
        bridge.cuda()
        bridge.eval()
        
        print(f"✓ GeometricBridge initialized")
        print(f"  - Bridge parameters: {sum(p.numel() for p in bridge.parameters() if p.requires_grad):,}")
        print_memory_stats()
        
        # Test forward pass
        dummy_image = torch.randn(1, 3, 224, 224).cuda()
        
        with torch.no_grad():
            output = bridge(dummy_image, return_uncertainty=True)
            
        print(f"✓ Forward pass successful")
        print(f"  - Adapted features shape: {output['adapted_features'].shape}")
        print(f"  - Uncertainty: {output['uncertainty'].item():.4f}")
        
        return bridge


def test_tta_adapter(bridge, config):
    """Test TestTimeAdapter functionality."""
    print("\nTesting TestTimeAdapter...")
    
    with MemoryProfiler("TTA Adapter"):
        tta_adapter = TestTimeAdapter(bridge, config)
        
        # Test single sample adaptation
        dummy_image = torch.randn(3, 224, 224).cuda()
        
        adapted_features, uncertainty = tta_adapter.adapt_single_sample(dummy_image)
        
        print(f"✓ TTA adaptation successful")
        print(f"  - Adapted features shape: {adapted_features.shape}")
        print(f"  - Uncertainty: {uncertainty:.4f}")
        
        # Test batch adaptation
        dummy_batch = torch.randn(4, 3, 224, 224).cuda()
        batch_features, batch_uncertainties = tta_adapter.adapt_batch(dummy_batch)
        
        print(f"✓ Batch TTA successful")
        print(f"  - Batch features shape: {batch_features.shape}")
        print(f"  - Mean uncertainty: {batch_uncertainties.mean():.4f}")
        
        return tta_adapter


def test_real_image_inference(bridge, tta_adapter, preprocess):
    """Test inference on a real (or dummy) image."""
    print("\nTesting real image inference...")
    
    # Load sample image
    image = load_sample_image()
    image_tensor = preprocess(image).unsqueeze(0).cuda()
    
    # Standard inference
    with torch.no_grad():
        standard_output = bridge(image_tensor, return_uncertainty=True)
    
    # TTA inference
    tta_features, tta_uncertainty = tta_adapter.adapt_single_sample(image_tensor.squeeze(0))
    
    print(f"✓ Real image inference successful")
    print(f"  - Standard uncertainty: {standard_output['uncertainty'].item():.4f}")
    print(f"  - TTA uncertainty: {tta_uncertainty:.4f}")
    
    # Compare features
    feature_similarity = torch.cosine_similarity(
        standard_output['adapted_features'],
        tta_features.unsqueeze(0),
        dim=-1
    ).item()
    
    print(f"  - Feature similarity: {feature_similarity:.4f}")
    
    return {
        'standard_uncertainty': standard_output['uncertainty'].item(),
        'tta_uncertainty': tta_uncertainty,
        'feature_similarity': feature_similarity
    }


def test_memory_scaling():
    """Test memory usage with different batch sizes."""
    print("\nTesting memory scaling...")
    
    config = create_dummy_config()
    bridge = GeometricBridge(config).cuda()
    bridge.eval()
    
    batch_sizes = [1, 2, 4, 8]
    
    print(f"{'Batch Size':<12} {'Memory (GB)':<12} {'Status'}")
    print("-" * 36)
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            
            dummy_batch = torch.randn(batch_size, 3, 224, 224).cuda()
            
            with torch.no_grad():
                _ = bridge(dummy_batch, return_uncertainty=True)
            
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            status = "✓"
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                memory_gb = "OOM"
                status = "✗"
            else:
                raise e
        
        print(f"{batch_size:<12} {memory_gb:<12} {status}")


def run_comprehensive_demo():
    """Run comprehensive demo of all functionality."""
    print("="*60)
    print("GeoTTA Comprehensive Demo")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This demo requires GPU.")
        return
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    print_memory_stats()
    
    # Create config
    config = create_dummy_config()
    print("✓ Configuration created")
    
    try:
        # Test 1: CLIP loading
        clip_model, preprocess = test_clip_loading()
        
        # Test 2: GeometricBridge
        bridge = test_geometric_bridge(config)
        
        # Test 3: TTA Adapter
        tta_adapter = test_tta_adapter(bridge, config)
        
        # Test 4: Real image inference
        results = test_real_image_inference(bridge, tta_adapter, preprocess)
        
        # Test 5: Memory scaling
        test_memory_scaling()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("Summary:")
        print(f"  - Standard Uncertainty: {results['standard_uncertainty']:.4f}")
        print(f"  - TTA Uncertainty: {results['tta_uncertainty']:.4f}")
        print(f"  - Feature Similarity: {results['feature_similarity']:.4f}")
        print_memory_stats()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up memory
        torch.cuda.empty_cache()


def create_simple_training_demo():
    """Demonstrate a simple training loop."""
    print("\n" + "="*60)
    print("Simple Training Demo")
    print("="*60)
    
    config = create_dummy_config()
    bridge = GeometricBridge(config).cuda()
    
    # Create dummy data
    num_samples = 16
    dummy_images = torch.randn(num_samples, 3, 224, 224).cuda()
    dummy_labels = torch.randint(0, 10, (num_samples,)).cuda()
    
    # Simple optimizer
    optimizer = torch.optim.Adam(
        [p for p in bridge.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    print(f"Training on {num_samples} dummy samples...")
    
    bridge.train()
    for epoch in range(3):  # Just 3 epochs for demo
        optimizer.zero_grad()
        
        # Forward pass
        output = bridge(dummy_images, return_uncertainty=True)
        
        # Dummy loss (just for demonstration)
        features = output['adapted_features']
        uncertainty = output['uncertainty']
        
        # Simple MSE loss
        target_features = torch.randn_like(features)
        loss = torch.nn.functional.mse_loss(features, target_features)
        
        # Add uncertainty regularization
        uncertainty_loss = uncertainty.mean()
        total_loss = loss + 0.1 * uncertainty_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f"  Epoch {epoch+1}: Loss = {total_loss.item():.4f}")
    
    print("✓ Training demo completed")
    bridge.eval()


if __name__ == '__main__':
    print("Starting GeoTTA Demo...")
    
    # Run main demo
    run_comprehensive_demo()
    
    # Optional: Run training demo
    try:
        create_simple_training_demo()
    except Exception as e:
        print(f"Training demo failed: {e}")
    
    print("\nDemo completed!")