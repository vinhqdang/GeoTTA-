import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
from tqdm import tqdm
from typing import Dict, Any

# Import GeoTTA modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from geotta.models.geometric_bridge import GeometricBridge, compute_logits_from_features
from geotta.models.uncertainty import GeometricUncertainty
from geotta.data.datasets import get_dataloader, get_text_dataloader
from geotta.utils.metrics import compute_metrics
from geotta.utils.memory import optimize_memory


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, scaler: GradScaler, 
                config: Dict[str, Any], epoch: int, writer: SummaryWriter) -> float:
    """Single training epoch with memory optimization."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Loss components
    ce_loss_fn = nn.CrossEntropyLoss()
    uncertainty_module = GeometricUncertainty(config)
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
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
            if output['original_text_features'] is not None:
                geo_uncertainty = uncertainty_module.compute_uncertainty(
                    output['original_image_features'],
                    output['original_text_features']
                )
                consistency_loss = F.mse_loss(
                    output['uncertainty'].squeeze(),
                    geo_uncertainty
                )
            else:
                consistency_loss = torch.tensor(0.0, device=images.device)
            
            # Total loss
            loss = cls_loss + 0.5 * calib_loss + 0.3 * consistency_loss
            
        # Backward pass with gradient accumulation
        scaler.scale(loss / config['training']['grad_accum_steps']).backward()
        
        if (batch_idx + 1) % config['training']['grad_accum_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'cls_loss': cls_loss.item(),
            'calib_loss': calib_loss.item(),
            'consistency_loss': consistency_loss.item()
        })
        
        # Log to tensorboard
        global_step = epoch * num_batches + batch_idx
        if batch_idx % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/cls_loss', cls_loss.item(), global_step)
            writer.add_scalar('train/calib_loss', calib_loss.item(), global_step)
            writer.add_scalar('train/consistency_loss', consistency_loss.item(), global_step)
        
        # Memory cleanup every 100 batches
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            
    return total_loss / len(dataloader)


def validate_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                  config: Dict[str, Any]) -> Dict[str, float]:
    """Validation epoch."""
    model.eval()
    all_preds = []
    all_labels = []
    all_uncertainties = []
    total_loss = 0
    
    ce_loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.cuda(), labels.cuda()
            
            output = model(images, return_uncertainty=True)
            logits = compute_logits_from_features(
                output['adapted_features'],
                model.get_text_prototypes()
            )
            
            loss = ce_loss_fn(logits / model.temperature, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            uncertainties = output['uncertainty'].squeeze()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_uncertainties.append(uncertainties.cpu())
    
    # Concatenate all predictions
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_uncertainties = torch.cat(all_uncertainties)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels, all_uncertainties)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer with weight decay."""
    # Only optimize trainable parameters (bridge components)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, 
                    config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['learning_rate'] * 0.01
    )
    
    return scheduler


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, config: Dict[str, Any], 
                   checkpoint_dir: str):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train GeoTTA Bridge')
    parser.add_argument('--config', default='geotta/configs/default.yaml', 
                       help='Path to config file')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', default='./logs', 
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', default=None, 
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Initialize model
    print("Initializing model...")
    model = GeometricBridge(config).cuda()
    
    # Memory optimization
    if config['training']['mixed_precision']:
        print("Using mixed precision training")
        scaler = GradScaler()
    else:
        scaler = None
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # Create tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train epoch
        train_loss = train_epoch(model, train_loader, optimizer, scaler, 
                               config, epoch, writer)
        
        # Validate epoch
        val_metrics = validate_epoch(model, val_loader, config)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, "
              f"val_acc={val_metrics['accuracy']:.4f}")
        
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, train_loss, 
                          config, args.checkpoint_dir)
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    writer.close()


if __name__ == '__main__':
    main()