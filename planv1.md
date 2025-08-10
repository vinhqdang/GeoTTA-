
TTA Implementation Plan v1.0
## Geometric Test-Time Adapter: Single-Pass Uncertainty-Aware Adaptation

### Project Overview
Implement a lightweight geometric adapter for test-time adaptation of CLIP models that provides uncertainty quantification through cross-modal geometric relationships.

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Project Setup
```bash
project_root/
├── geotta/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── geometric_bridge.py      # Core geometric adapter
│   │   ├── uncertainty.py           # Uncertainty computation
│   │   └── hyperbolic.py           # Hyperbolic geometry utils
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py             # Dataset loaders
│   │   └── augmentations.py        # Test-time augmentations
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Evaluation metrics
│   │   ├── visualization.py        # Plotting utilities
│   │   └── memory.py               # Memory optimization utils
│   └── configs/
│       └── default.yaml            # Configuration file
├── scripts/
│   ├── train_bridge.py             # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── demo_tta.py                 # Demo for test-time adaptation
├── requirements.txt
└── README.md
```

#### 1.2 Dependencies
```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
clip @ git+https://github.com/openai/CLIP.git
timm>=0.9.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
yaml>=0.2.5
tensorboard>=2.13.0
einops>=0.7.0
geoopt>=0.5.0  # For hyperbolic geometry
```

#### 1.3 Configuration File
```yaml
# configs/default.yaml
model:
  clip_model: "ViT-B/32"  # Start with smallest CLIP
  bridge_dim: 512
  bridge_layers: 2
  bridge_heads: 8
  dropout: 0.1
  use_hyperbolic: true
  temperature: 0.07

training:
  batch_size: 16  # Will use gradient accumulation
  grad_accum_steps: 4  # Effective batch = 64
  learning_rate: 1e-4
  weight_decay: 0.01
  epochs: 30
  warmup_steps: 500
  
  # Memory optimization
  mixed_precision: true
  gradient_checkpointing: false  # Only if needed
  
data:
  train_dataset: "imagenet_subset"  # Start with subset
  val_dataset: "imagenet_val"
  num_workers: 4
  pin_memory: true
  
uncertainty:
  geometric_weight: 1.0
  angular_weight: 0.5
  calibration_bins: 15

test_time:
  adaptation_steps: 1  # Single-pass
  adaptation_lr: 0.001
  cache_size: 32  # For K-NN style adaptation
```

### Phase 2: Core Components Implementation

#### 2.1 Geometric Bridge Module
```python
# geotta/models/geometric_bridge.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import clip

class GeometricBridge(nn.Module):
    """
    Lightweight geometric adapter between CLIP modalities.
    Learns to predict and correct geometric transformations.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.model.bridge_dim
        
        # Frozen CLIP encoders
        self.clip_model, _ = clip.load(config.model.clip_model, device='cuda')
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Geometric transformation layers (only these are trainable)
        self.image_proj = nn.Linear(512, dim)  # CLIP ViT-B/32 outputs 512d
        self.text_proj = nn.Linear(512, dim)
        
        # Cross-modal attention for geometric alignment
        self.cross_attention = nn.MultiheadAttention(
            dim, 
            config.model.bridge_heads,
            dropout=config.model.dropout,
            batch_first=True
        )
        
        # Geometric uncertainty predictor
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim, 512)
        
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * config.model.temperature)
        
    def forward(self, images, texts=None, return_uncertainty=True):
        """
        Forward pass computing adapted features and uncertainty.
        
        Args:
            images: Batch of images [B, 3, H, W]
            texts: Optional text tokens for training
            return_uncertainty: Whether to compute uncertainty
        """
        # Extract frozen CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            if texts is not None:
                text_features = self.clip_model.encode_text(texts)
                text_features = F.normalize(text_features, p=2, dim=-1)
            else:
                # During test-time, use cached prototypes
                text_features = self.get_cached_prototypes()
        
        # Project to bridge space
        img_bridge = self.image_proj(image_features)
        txt_bridge = self.text_proj(text_features)
        
        # Compute cross-modal attention (geometric alignment)
        aligned_img, attention_weights = self.cross_attention(
            img_bridge.unsqueeze(1),
            txt_bridge.unsqueeze(1),
            txt_bridge.unsqueeze(1)
        )
        aligned_img = aligned_img.squeeze(1)
        
        # Compute geometric uncertainty
        if return_uncertainty:
            # Concatenate original and aligned features
            uncertainty_input = torch.cat([img_bridge, aligned_img], dim=-1)
            uncertainty = self.uncertainty_head(uncertainty_input)
        else:
            uncertainty = None
            
        # Project back to CLIP space
        adapted_features = self.output_proj(aligned_img)
        adapted_features = F.normalize(adapted_features, p=2, dim=-1)
        
        return {
            'adapted_features': adapted_features,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'original_image_features': image_features,
            'original_text_features': text_features if texts is not None else None
        }
```

#### 2.2 Uncertainty Computation Module
```python
# geotta/models/uncertainty.py

import torch
import torch.nn.functional as F
import numpy as np

class GeometricUncertainty:
    """
    Compute uncertainty based on geometric relationships in embedding space.
    """
    def __init__(self, config):
        self.config = config
        self.geometric_weight = config.uncertainty.geometric_weight
        self.angular_weight = config.uncertainty.angular_weight
        
    def compute_uncertainty(self, image_features, text_features):
        """
        Compute uncertainty from cross-modal geometric distance.
        
        Args:
            image_features: [B, D] normalized image embeddings
            text_features: [B, D] or [K, D] normalized text embeddings
        """
        # Euclidean distance in normalized space
        if text_features.dim() == 3:  # [B, K, D] - multiple prototypes per image
            distances = torch.cdist(
                image_features.unsqueeze(1), 
                text_features, 
                p=2
            )
            min_distances, _ = distances.min(dim=1)
        else:
            distances = torch.norm(image_features - text_features, p=2, dim=-1)
            min_distances = distances
            
        # Angular distance (more robust to modality gap)
        cosine_sim = F.cosine_similarity(image_features, text_features, dim=-1)
        angular_distance = torch.acos(torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7))
        
        # Combined uncertainty
        uncertainty = (self.geometric_weight * min_distances + 
                      self.angular_weight * angular_distance)
        
        return uncertainty
    
    def calibrate_uncertainty(self, uncertainties, errors, num_bins=15):
        """
        Calibrate uncertainty to actual errors (for training).
        """
        # Compute calibration error
        uncertainties_np = uncertainties.cpu().numpy()
        errors_np = errors.cpu().numpy()
        
        # Bin uncertainties
        bin_boundaries = np.linspace(0, uncertainties_np.max(), num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0  # Expected Calibration Error
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties_np > bin_lower) & (uncertainties_np <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_uncertainty_in_bin = uncertainties_np[in_bin].mean()
                avg_error_in_bin = errors_np[in_bin].mean()
                ece += np.abs(avg_uncertainty_in_bin - avg_error_in_bin) * prop_in_bin
                
        return ece
```

#### 2.3 Test-Time Adaptation Module
```python
# geotta/models/tta_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class TestTimeAdapter:
    """
    Efficient test-time adaptation using geometric uncertainty.
    No backpropagation required during test time.
    """
    def __init__(self, bridge_model, config):
        self.bridge = bridge_model
        self.config = config
        
        # Cache for prototypes
        self.prototype_cache = deque(maxlen=config.test_time.cache_size)
        self.uncertainty_cache = deque(maxlen=config.test_time.cache_size)
        
        # Running statistics for calibration
        self.running_mean_uncertainty = 0
        self.running_var_uncertainty = 0
        self.num_samples = 0
        
    @torch.no_grad()
    def adapt_single_sample(self, image):
        """
        Single-pass adaptation for one test sample.
        """
        # Get features and uncertainty
        output = self.bridge(image.unsqueeze(0), return_uncertainty=True)
        
        adapted_features = output['adapted_features']
        uncertainty = output['uncertainty'].item()
        
        # Update running statistics
        self.update_statistics(uncertainty)
        
        # Compute adaptation weight based on normalized uncertainty
        norm_uncertainty = self.normalize_uncertainty(uncertainty)
        adaptation_weight = torch.sigmoid(-norm_uncertainty)  # High confidence = high weight
        
        # Weighted combination with original features
        final_features = (adaptation_weight * adapted_features + 
                         (1 - adaptation_weight) * output['original_image_features'])
        
        # Update cache if confidence is high
        if adaptation_weight > 0.7:
            self.prototype_cache.append(final_features)
            self.uncertainty_cache.append(uncertainty)
            
        return final_features, uncertainty
    
    def update_statistics(self, uncertainty):
        """Update running statistics for uncertainty normalization."""
        self.num_samples += 1
        delta = uncertainty - self.running_mean_uncertainty
        self.running_mean_uncertainty += delta / self.num_samples
        self.running_var_uncertainty += delta * (uncertainty - self.running_mean_uncertainty)
        
    def normalize_uncertainty(self, uncertainty):
        """Normalize uncertainty using running statistics."""
        if self.num_samples < 2:
            return uncertainty
        
        std = (self.running_var_uncertainty / (self.num_samples - 1)) ** 0.5
        if std < 1e-6:
            return 0
        
        return (uncertainty - self.running_mean_uncertainty) / std
```

### Phase 3: Training Pipeline

#### 3.1 Training Script Structure
```python
# scripts/train_bridge.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

from geotta.models.geometric_bridge import GeometricBridge
from geotta.models.uncertainty import GeometricUncertainty
from geotta.data.datasets import get_dataloader
from geotta.utils.metrics import compute_metrics
from geotta.utils.memory import optimize_memory

def train_epoch(model, dataloader, optimizer, scaler, config):
    """Single training epoch with memory optimization."""
    model.train()
    total_loss = 0
    
    # Loss components
    ce_loss_fn = nn.CrossEntropyLoss()
    uncertainty_module = GeometricUncertainty(config)
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.cuda(), labels.cuda()
        
        # Mixed precision training
        with autocast():
            # Forward pass
            output = model(images, return_uncertainty=True)
            
            # Compute logits using adapted features
            logits = compute_logits_from_features(
                output['adapted_features'],
                model.get_text_prototypes()
            )
            
            # Classification loss
            cls_loss = ce_loss_fn(logits / model.temperature, labels)
            
            # Uncertainty calibration loss
            pred_probs = F.softmax(logits / model.temperature, dim=-1)
            pred_uncertainty = 1 - pred_probs.max(dim=-1)[0]
            
            calib_loss = F.mse_loss(
                output['uncertainty'].squeeze(),
                pred_uncertainty
            )
            
            # Geometric consistency loss
            geo_uncertainty = uncertainty_module.compute_uncertainty(
                output['original_image_features'],
                output['original_text_features']
            )
            consistency_loss = F.mse_loss(
                output['uncertainty'].squeeze(),
                geo_uncertainty
            )
            
            # Total loss
            loss = cls_loss + 0.5 * calib_loss + 0.3 * consistency_loss
            
        # Backward pass with gradient accumulation
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.training.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        
        # Memory cleanup every 100 batches
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            
    return total_loss / len(dataloader)

def compute_logits_from_features(image_features, text_features):
    """Compute classification logits from features."""
    # Simple dot product in CLIP space
    logits = 100.0 * image_features @ text_features.T
    return logits
```

#### 3.2 Data Loading with Memory Optimization
```python
# geotta/data/datasets.py

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import clip

class CLIPDataset(Dataset):
    """Memory-efficient dataset for CLIP training."""
    
    def __init__(self, root, split='train', config=None):
        self.root = root
        self.split = split
        self.config = config
        
        # Load image paths instead of images to save memory
        self.image_paths = self.load_image_paths()
        self.labels = self.load_labels()
        
        # CLIP preprocessing
        _, self.preprocess = clip.load(config.model.clip_model, device='cpu')
        
        # Additional augmentations for training
        if split == 'train':
            self.transform = transforms.Compose([
                self.preprocess,
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = self.preprocess
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image on-demand to save memory
        image = self.load_image(self.image_paths[idx])
        image = self.transform(image)
        label = self.labels[idx]
        
        return image, label
    
    def load_image(self, path):
        """Load and preprocess a single image."""
        from PIL import Image
        return Image.open(path).convert('RGB')

def get_dataloader(config, split='train'):
    """Get memory-optimized dataloader."""
    dataset = CLIPDataset(
        root=config.data.root,
        split=split,
        config=config
    )
    
    # Small batch size for 8GB VRAM
    batch_size = config.training.batch_size if split == 'train' else 1
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=(split == 'train'),
        persistent_workers=True  # Keep workers alive to reduce overhead
    )
    
    return dataloader
```

### Phase 4: Evaluation and Testing

#### 4.1 Evaluation Metrics
```python
# geotta/utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, calibration_curve

def evaluate_model(model, dataloader, config):
    """Comprehensive evaluation including uncertainty quality."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            
            output = model(images, return_uncertainty=True)
            logits = compute_logits_from_features(
                output['adapted_features'],
                model.get_text_prototypes()
            )
            
            preds = logits.argmax(dim=-1)
            uncertainties = output['uncertainty'].squeeze()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_uncertainties.append(uncertainties.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_uncertainties = torch.cat(all_uncertainties)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Uncertainty calibration
    errors = (all_preds != all_labels).float()
    ece = expected_calibration_error(all_uncertainties, errors)
    
    # Uncertainty quality (AUROC for error detection)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(errors, all_uncertainties)
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'auroc': auroc
    }
```

### Phase 5: Quick Start Implementation

#### 5.1 Minimal Working Example
```python
# demo_minimal.py
"""
Minimal example to test GeoTTA with 8GB VRAM.
Start with this to verify everything works.
"""

import torch
import clip
from geotta.models.geometric_bridge import GeometricBridge

# Load configuration
config = {
    'model': {
        'clip_model': 'ViT-B/32',
        'bridge_dim': 256,  # Smaller for testing
        'bridge_layers': 1,
        'bridge_heads': 4,
        'dropout': 0.1,
        'temperature': 0.07
    }
}

# Initialize model
model = GeometricBridge(config)
model.cuda()

# Test with single image
from PIL import Image
image = Image.open('test_image.jpg')
_, preprocess = clip.load('ViT-B/32')
image_tensor = preprocess(image).unsqueeze(0).cuda()

# Forward pass
with torch.no_grad():
    output = model(image_tensor)
    print(f"Adapted features shape: {output['adapted_features'].shape}")
    print(f"Uncertainty: {output['uncertainty'].item():.4f}")
```

### Implementation Timeline

**Week 1: Core Infrastructure**
- Set up project structure
- Implement GeometricBridge module
- Test forward pass with dummy data
- Verify memory usage stays under 8GB

**Week 2: Training Pipeline**
- Implement training loop with gradient accumulation
- Add uncertainty computation
- Set up data loaders for ImageNet subset
- Run first training experiments

**Week 3: Test-Time Adaptation**
- Implement single-pass adaptation
- Add prototype caching mechanism
- Test on distribution shift datasets
- Optimize for inference speed

**Week 4: Evaluation & Optimization**
- Comprehensive evaluation on benchmarks
- Memory profiling and optimization
- Hyperparameter tuning
- Documentation and demo preparation

### Memory Optimization Checklist

- [ ] Use mixed precision training (fp16)
- [ ] Implement gradient accumulation (effective batch size)
- [ ] Freeze CLIP encoders (no gradients)
- [ ] Load images on-demand in dataset
- [ ] Clear cache periodically during training
- [ ] Use gradient checkpointing if needed
- [ ] Keep adapter under 10M parameters
- [ ] Use smaller CLIP model (ViT-B/32) initially
- [ ] Pin memory for faster transfer
- [ ] Persistent workers in DataLoader

### Testing Strategy

1. **Unit Tests**: Each module independently
2. **Memory Tests**: Ensure < 8GB usage
3. **Speed Tests**: Single-pass inference < 50ms
4. **Accuracy Tests**: Match baseline CLIP on clean data
5. **Uncertainty Tests**: Calibration on OOD data

### Expected Outputs

After Week 1:
- Working forward pass
- Memory usage confirmation

After Week 2:
- Training curves showing convergence
- Initial accuracy metrics

After Week 3:
- Test-time adaptation working
- Uncertainty estimates calibrated

After Week 4:
- Full evaluation results
- Comparison with baselines
- Ready for paper experiments

### Notes for Implementation

1. Start with smallest CLIP model (ViT-B/32)
2. Use ImageNet subset (e.g., 100 classes) for initial experiments
3. Monitor GPU memory constantly
4. Use tensorboard for tracking experiments
5. Save checkpoints frequently
6. Test on CPU first if GPU OOM occurs

This plan provides a complete roadmap for implementing GeoTTA v1.0 with your hardware constraints.
