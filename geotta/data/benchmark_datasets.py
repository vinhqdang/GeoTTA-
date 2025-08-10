import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import clip
import os
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
import requests
import tarfile
import zipfile
from pathlib import Path


class BenchmarkDatasetManager:
    """
    Comprehensive dataset manager for WACV 2026 experiments.
    Supports all major vision benchmarks and domain shift datasets.
    """
    
    DATASETS = {
        # Standard benchmarks
        'imagenet': {
            'name': 'ImageNet-1K',
            'num_classes': 1000,
            'splits': ['train', 'val'],
            'download_url': None,  # Requires manual download
        },
        'cifar10': {
            'name': 'CIFAR-10', 
            'num_classes': 10,
            'splits': ['train', 'test'],
            'download_url': 'auto',  # torchvision handles
        },
        'cifar100': {
            'name': 'CIFAR-100',
            'num_classes': 100, 
            'splits': ['train', 'test'],
            'download_url': 'auto',
        },
        
        # Domain adaptation benchmarks
        'office_home': {
            'name': 'Office-Home',
            'num_classes': 65,
            'domains': ['Art', 'Clipart', 'Product', 'RealWorld'],
            'download_url': 'https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg',
        },
        'domainnet': {
            'name': 'DomainNet',
            'num_classes': 345,
            'domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
            'download_url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/',
        },
        
        # Robustness benchmarks  
        'imagenet_c': {
            'name': 'ImageNet-C',
            'num_classes': 1000,
            'corruptions': [
                'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
            ],
            'severities': [1, 2, 3, 4, 5],
            'download_url': 'https://zenodo.org/record/2235448/files/ImageNet-C.tar',
        },
        'imagenet_r': {
            'name': 'ImageNet-R', 
            'num_classes': 200,
            'download_url': 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar',
        },
        'imagenet_a': {
            'name': 'ImageNet-A',
            'num_classes': 200, 
            'download_url': 'https://people.eecs.berkeley.edu/~hendrycks/natural.tar',
        },
        'imagenet_v2': {
            'name': 'ImageNet-V2',
            'num_classes': 1000,
            'download_url': 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz',
        },
        
        # Few-shot benchmarks
        'caltech101': {
            'name': 'Caltech-101',
            'num_classes': 101,
            'download_url': 'auto',
        },
        'oxford_pets': {
            'name': 'Oxford-IIIT Pets',
            'num_classes': 37,
            'download_url': 'auto',
        },
        'stanford_cars': {
            'name': 'Stanford Cars',
            'num_classes': 196,
            'download_url': 'auto',
        },
        'flowers102': {
            'name': 'Oxford Flowers-102',
            'num_classes': 102,
            'download_url': 'auto',
        },
        'food101': {
            'name': 'Food-101',
            'num_classes': 101,
            'download_url': 'auto',
        },
        'fgvc_aircraft': {
            'name': 'FGVC Aircraft',
            'num_classes': 100,
            'download_url': 'auto',
        },
        'sun397': {
            'name': 'SUN397',
            'num_classes': 397,
            'download_url': 'auto',
        },
        'dtd': {
            'name': 'Describable Textures (DTD)',
            'num_classes': 47,
            'download_url': 'auto',
        },
        'eurosat': {
            'name': 'EuroSAT',
            'num_classes': 10,
            'download_url': 'auto',
        },
        'ucf101': {
            'name': 'UCF-101',
            'num_classes': 101,
            'download_url': 'auto',
        },
    }
    
    def __init__(self, root_dir: str = './data'):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)
        
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.DATASETS[dataset_name]
    
    def download_dataset(self, dataset_name: str, force_download: bool = False):
        """Download a dataset if needed."""
        info = self.get_dataset_info(dataset_name)
        dataset_dir = self.root_dir / dataset_name
        
        if dataset_dir.exists() and not force_download:
            print(f"Dataset {dataset_name} already exists at {dataset_dir}")
            return
        
        dataset_dir.mkdir(exist_ok=True)
        
        if info['download_url'] == 'auto':
            self._download_torchvision_dataset(dataset_name, dataset_dir)
        elif info['download_url']:
            self._download_from_url(info['download_url'], dataset_dir)
        else:
            print(f"Manual download required for {dataset_name}")
            print(f"Please download to: {dataset_dir}")
    
    def _download_torchvision_dataset(self, dataset_name: str, dataset_dir: Path):
        """Download datasets available in torchvision."""
        transform = transforms.ToTensor()
        
        if dataset_name == 'cifar10':
            datasets.CIFAR10(str(dataset_dir), train=True, download=True, transform=transform)
            datasets.CIFAR10(str(dataset_dir), train=False, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            datasets.CIFAR100(str(dataset_dir), train=True, download=True, transform=transform)
            datasets.CIFAR100(str(dataset_dir), train=False, download=True, transform=transform)
        elif dataset_name == 'caltech101':
            datasets.Caltech101(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'oxford_pets':
            datasets.OxfordIIITPet(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'stanford_cars':
            datasets.StanfordCars(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'flowers102':
            datasets.Flowers102(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'food101':
            datasets.Food101(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'fgvc_aircraft':
            datasets.FGVCAircraft(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'sun397':
            datasets.SUN397(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'dtd':
            datasets.DTD(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'eurosat':
            datasets.EuroSAT(str(dataset_dir), download=True, transform=transform)
        elif dataset_name == 'ucf101':
            datasets.UCF101(str(dataset_dir), download=True, transform=transform)
        else:
            raise ValueError(f"Unknown torchvision dataset: {dataset_name}")
    
    def _download_from_url(self, url: str, dataset_dir: Path):
        """Download and extract dataset from URL."""
        print(f"Downloading from {url}")
        # Implementation would depend on the specific dataset format
        # This is a placeholder for the actual download logic
        pass
    
    def get_dataset(self, dataset_name: str, split: str = 'test', 
                   clip_model: str = 'ViT-B/32', **kwargs) -> Dataset:
        """Get a dataset instance."""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_dir = self.root_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Dataset {dataset_name} not found. Attempting download...")
            self.download_dataset(dataset_name)
        
        return self._create_dataset_instance(dataset_name, split, clip_model, **kwargs)
    
    def _create_dataset_instance(self, dataset_name: str, split: str, 
                               clip_model: str, **kwargs) -> Dataset:
        """Create the appropriate dataset instance."""
        if dataset_name in ['cifar10', 'cifar100']:
            return CIFARDataset(dataset_name, self.root_dir, split, clip_model)
        elif dataset_name == 'imagenet':
            return ImageNetDataset(self.root_dir, split, clip_model)
        elif dataset_name == 'imagenet_c':
            return ImageNetCDataset(self.root_dir, split, clip_model, **kwargs)
        elif dataset_name in ['imagenet_r', 'imagenet_a', 'imagenet_v2']:
            return ImageNetVariantDataset(dataset_name, self.root_dir, clip_model)
        elif dataset_name == 'office_home':
            return OfficeHomeDataset(self.root_dir, split, clip_model, **kwargs)
        elif dataset_name == 'domainnet':
            return DomainNetDataset(self.root_dir, split, clip_model, **kwargs)
        else:
            return TorchvisionDataset(dataset_name, self.root_dir, split, clip_model)


class BaseVisionDataset(Dataset):
    """Base class for vision datasets with CLIP preprocessing."""
    
    def __init__(self, dataset_name: str, root_dir: Path, split: str, clip_model: str):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.split = split
        self.clip_model = clip_model
        
        # Load CLIP preprocessing
        _, self.preprocess = clip.load(clip_model, device='cpu')
        
        # Load data
        self.samples = self._load_samples()
        self.class_names = self._get_class_names()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load samples as (image_path, label) tuples."""
        raise NotImplementedError
    
    def _get_class_names(self) -> List[str]:
        """Get class names for text prompts."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        image_path, label = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image)
        
        # Metadata for analysis
        metadata = {
            'image_path': str(image_path),
            'class_name': self.class_names[label],
            'dataset': self.dataset_name,
            'split': self.split
        }
        
        return image_tensor, label, metadata


class CIFARDataset(BaseVisionDataset):
    """CIFAR-10/100 dataset wrapper."""
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        is_train = (self.split == 'train')
        
        if self.dataset_name == 'cifar10':
            dataset = datasets.CIFAR10(
                str(self.root_dir / 'cifar10'), 
                train=is_train, 
                download=False
            )
        else:  # cifar100
            dataset = datasets.CIFAR100(
                str(self.root_dir / 'cifar100'), 
                train=is_train, 
                download=False
            )
        
        # Save images to disk for consistent interface
        samples = []
        split_dir = self.root_dir / self.dataset_name / self.split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (image, label) in enumerate(dataset):
            image_path = split_dir / f"{i:06d}_{label}.png"
            if not image_path.exists():
                image.save(image_path)
            samples.append((str(image_path), label))
        
        return samples
    
    def _get_class_names(self) -> List[str]:
        if self.dataset_name == 'cifar10':
            return [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        else:  # cifar100
            return [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]


class ImageNetDataset(BaseVisionDataset):
    """ImageNet dataset wrapper."""
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        dataset_dir = self.root_dir / 'imagenet' / self.split
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"ImageNet {self.split} not found at {dataset_dir}")
        
        samples = []
        class_dirs = sorted(dataset_dir.glob('*/'))
        
        for class_idx, class_dir in enumerate(class_dirs):
            for image_path in class_dir.glob('*.JPEG'):
                samples.append((str(image_path), class_idx))
        
        return samples
    
    def _get_class_names(self) -> List[str]:
        # Load ImageNet class names
        imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        try:
            response = requests.get(imagenet_classes_url)
            class_names = response.text.strip().split('\n')
        except:
            # Fallback to generic names
            class_names = [f'class_{i}' for i in range(1000)]
        
        return class_names


class ImageNetCDataset(BaseVisionDataset):
    """ImageNet-C (corruption) dataset wrapper."""
    
    def __init__(self, root_dir: Path, split: str, clip_model: str, 
                 corruption: str = 'gaussian_noise', severity: int = 1):
        self.corruption = corruption
        self.severity = severity
        super().__init__('imagenet_c', root_dir, split, clip_model)
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        dataset_dir = self.root_dir / 'imagenet_c' / self.corruption / str(self.severity)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"ImageNet-C data not found at {dataset_dir}")
        
        samples = []
        for image_path in sorted(dataset_dir.glob('*.JPEG')):
            # Extract label from filename or directory structure
            label = self._extract_label_from_path(image_path)
            samples.append((str(image_path), label))
        
        return samples
    
    def _extract_label_from_path(self, image_path: Path) -> int:
        # Implementation depends on ImageNet-C structure
        # This is a placeholder
        return 0
    
    def _get_class_names(self) -> List[str]:
        # Same as ImageNet
        imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        try:
            response = requests.get(imagenet_classes_url)
            class_names = response.text.strip().split('\n')
        except:
            class_names = [f'class_{i}' for i in range(1000)]
        
        return class_names


class ImageNetVariantDataset(BaseVisionDataset):
    """ImageNet variants (R, A, V2) dataset wrapper."""
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        dataset_dir = self.root_dir / self.dataset_name
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"{self.dataset_name} not found at {dataset_dir}")
        
        samples = []
        if self.dataset_name in ['imagenet_r', 'imagenet_a']:
            # These have class subdirectories
            class_dirs = sorted(dataset_dir.glob('*/'))
            for class_idx, class_dir in enumerate(class_dirs):
                for image_path in class_dir.glob('*.JPEG'):
                    samples.append((str(image_path), class_idx))
        else:  # imagenet_v2
            # Flat structure with labels file
            for image_path in sorted(dataset_dir.glob('*.JPEG')):
                label = self._extract_label_from_path(image_path)
                samples.append((str(image_path), label))
        
        return samples
    
    def _extract_label_from_path(self, image_path: Path) -> int:
        # Implementation depends on specific dataset structure
        return 0
    
    def _get_class_names(self) -> List[str]:
        if self.dataset_name in ['imagenet_r', 'imagenet_a']:
            # 200 classes subset
            return [f'class_{i}' for i in range(200)]
        else:  # imagenet_v2
            # 1000 classes like original ImageNet
            return [f'class_{i}' for i in range(1000)]


class OfficeHomeDataset(BaseVisionDataset):
    """Office-Home dataset wrapper for domain adaptation."""
    
    def __init__(self, root_dir: Path, split: str, clip_model: str, 
                 domain: str = 'RealWorld'):
        self.domain = domain
        super().__init__('office_home', root_dir, split, clip_model)
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        dataset_dir = self.root_dir / 'office_home' / self.domain
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Office-Home {self.domain} not found at {dataset_dir}")
        
        samples = []
        class_dirs = sorted(dataset_dir.glob('*/'))
        
        for class_idx, class_dir in enumerate(class_dirs):
            for image_path in class_dir.glob('*.jpg'):
                samples.append((str(image_path), class_idx))
        
        return samples
    
    def _get_class_names(self) -> List[str]:
        # Office-Home has 65 classes
        return [f'class_{i}' for i in range(65)]


class DomainNetDataset(BaseVisionDataset):
    """DomainNet dataset wrapper for domain adaptation."""
    
    def __init__(self, root_dir: Path, split: str, clip_model: str, 
                 domain: str = 'real'):
        self.domain = domain
        super().__init__('domainnet', root_dir, split, clip_model)
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        dataset_dir = self.root_dir / 'domainnet' / self.domain
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"DomainNet {self.domain} not found at {dataset_dir}")
        
        samples = []
        class_dirs = sorted(dataset_dir.glob('*/'))
        
        for class_idx, class_dir in enumerate(class_dirs):
            for image_path in class_dir.glob('*.jpg'):
                samples.append((str(image_path), class_idx))
        
        return samples
    
    def _get_class_names(self) -> List[str]:
        # DomainNet has 345 classes
        return [f'class_{i}' for i in range(345)]


class TorchvisionDataset(BaseVisionDataset):
    """Generic wrapper for torchvision datasets."""
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        # This would be implemented based on the specific torchvision dataset
        # For now, return empty list as placeholder
        return []
    
    def _get_class_names(self) -> List[str]:
        # This would be implemented based on the specific dataset
        return []


def get_benchmark_dataloader(dataset_name: str, split: str = 'test', 
                           batch_size: int = 32, clip_model: str = 'ViT-B/32',
                           shuffle: bool = False, num_workers: int = 4,
                           root_dir: str = './data', **kwargs) -> DataLoader:
    """
    Get dataloader for benchmark dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split ('train', 'test', 'val')
        batch_size: Batch size
        clip_model: CLIP model variant
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        root_dir: Root directory for datasets
        **kwargs: Additional arguments for specific datasets
    
    Returns:
        DataLoader instance
    """
    manager = BenchmarkDatasetManager(root_dir)
    dataset = manager.get_dataset(dataset_name, split, clip_model, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0)
    )