import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import clip
from PIL import Image
import os
from typing import Dict, Any, Tuple, List


class CLIPDataset(Dataset):
    """Memory-efficient dataset for CLIP training."""
    
    def __init__(self, root: str, split: str = 'train', config: Dict[str, Any] = None):
        self.root = root
        self.split = split
        self.config = config
        
        # Load image paths instead of images to save memory
        self.image_paths, self.labels, self.class_names = self.load_data()
        
        # CLIP preprocessing
        _, self.preprocess = clip.load(config['model']['clip_model'], device='cpu')
        
        # Additional augmentations for training
        if split == 'train':
            self.transform = transforms.Compose([
                self.preprocess.transforms[0],  # Resize
                self.preprocess.transforms[1],  # CenterCrop
                transforms.RandomHorizontalFlip(p=0.5),
                self.preprocess.transforms[2],  # ToTensor
                self.preprocess.transforms[3],  # Normalize
            ])
        else:
            self.transform = self.preprocess
            
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image on-demand to save memory
        image = self.load_image(self.image_paths[idx])
        image = self.transform(image)
        label = self.labels[idx]
        
        return image, label
    
    def load_image(self, path: str) -> Image.Image:
        """Load and preprocess a single image."""
        return Image.open(path).convert('RGB')
    
    def load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load dataset paths and labels."""
        if self.config['data']['train_dataset'] == 'cifar10':
            return self.load_cifar10()
        elif self.config['data']['train_dataset'] == 'imagenet_subset':
            return self.load_imagenet_subset()
        else:
            # Generic dataset loading
            return self.load_generic_dataset()
    
    def load_cifar10(self) -> Tuple[List[str], List[int], List[str]]:
        """Load CIFAR-10 dataset."""
        # Use torchvision CIFAR-10 but extract paths
        dataset = datasets.CIFAR10(
            root=self.root, 
            train=(self.split == 'train'), 
            download=True
        )
        
        # CIFAR-10 class names
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # For CIFAR-10, we'll store images in memory since they're small
        image_paths = []
        labels = []
        
        for i, (image, label) in enumerate(dataset):
            # Save images temporarily
            temp_path = f'/tmp/cifar10_{self.split}_{i}.png'
            image.save(temp_path)
            image_paths.append(temp_path)
            labels.append(label)
        
        return image_paths, labels, class_names
    
    def load_imagenet_subset(self) -> Tuple[List[str], List[int], List[str]]:
        """Load a subset of ImageNet for testing."""
        # This is a placeholder - you'd need to implement based on your ImageNet setup
        # For now, we'll create a dummy dataset structure
        
        class_names = [f'class_{i}' for i in range(100)]  # 100 classes subset
        image_paths = []
        labels = []
        
        # Look for images in the root directory structure
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.root, self.split, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, img_file))
                        labels.append(class_idx)
        
        return image_paths, labels, class_names
    
    def load_generic_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """Load a generic dataset from directory structure."""
        image_paths = []
        labels = []
        class_names = []
        
        # Assume directory structure: root/split/class_name/image_files
        split_dir = os.path.join(self.root, self.split)
        if os.path.exists(split_dir):
            class_names = sorted(os.listdir(split_dir))
            
            for class_idx, class_name in enumerate(class_names):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(class_dir, img_file))
                            labels.append(class_idx)
        
        return image_paths, labels, class_names
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_names


class CLIPTextDataset(Dataset):
    """Dataset for text prompts corresponding to classes."""
    
    def __init__(self, class_names: List[str], templates: List[str] = None):
        self.class_names = class_names
        self.templates = templates or [
            "a photo of a {}.",
            "a photograph of a {}.",
            "an image of a {}.",
            "{} in the image.",
            "this is a {}.",
        ]
        
        # Generate all text prompts
        self.text_prompts = []
        self.labels = []
        
        for class_idx, class_name in enumerate(class_names):
            for template in self.templates:
                prompt = template.format(class_name)
                self.text_prompts.append(prompt)
                self.labels.append(class_idx)
    
    def __len__(self) -> int:
        return len(self.text_prompts)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.text_prompts[idx], self.labels[idx]


def get_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """Get memory-optimized dataloader."""
    
    # Create dataset
    if split in ['train', 'val']:
        dataset = CLIPDataset(
            root=config['data'].get('root', './data'),
            split=split,
            config=config
        )
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Small batch size for 8GB VRAM
    batch_size = config['training']['batch_size'] if split == 'train' else 1
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=(split == 'train'),
        persistent_workers=True  # Keep workers alive to reduce overhead
    )
    
    return dataloader


def get_text_dataloader(class_names: List[str], batch_size: int = 32) -> DataLoader:
    """Get dataloader for text prompts."""
    dataset = CLIPTextDataset(class_names)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Text processing is fast, no need for multiprocessing
    )
    
    return dataloader


def create_imagenet_subset(source_path: str, target_path: str, 
                          num_classes: int = 100, samples_per_class: int = 500):
    """Create a subset of ImageNet for training."""
    import shutil
    import random
    
    # Get list of all ImageNet classes
    all_classes = sorted(os.listdir(os.path.join(source_path, 'train')))
    selected_classes = random.sample(all_classes, num_classes)
    
    # Create target directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(target_path, split), exist_ok=True)
        
        for class_name in selected_classes:
            src_dir = os.path.join(source_path, split, class_name)
            dst_dir = os.path.join(target_path, split, class_name)
            
            if os.path.exists(src_dir):
                os.makedirs(dst_dir, exist_ok=True)
                
                # Copy subset of images
                all_images = [f for f in os.listdir(src_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                selected_images = random.sample(all_images, 
                                              min(samples_per_class, len(all_images)))
                
                for img_name in selected_images:
                    shutil.copy2(
                        os.path.join(src_dir, img_name),
                        os.path.join(dst_dir, img_name)
                    )
    
    print(f"Created ImageNet subset with {num_classes} classes at {target_path}")