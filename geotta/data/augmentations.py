import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import random
from typing import List, Tuple


class TestTimeAugmentation:
    """Test-time augmentation for improving model robustness."""
    
    def __init__(self, num_augmentations: int = 5):
        self.num_augmentations = num_augmentations
        self.base_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        ]
    
    def __call__(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Apply multiple random augmentations to the input image."""
        augmented_images = [image]  # Include original
        
        for _ in range(self.num_augmentations - 1):
            # Randomly select and apply transforms
            transform = random.choice(self.base_transforms)
            augmented = transform(image)
            augmented_images.append(augmented)
        
        return augmented_images


class GeometricAugmentations:
    """Geometric augmentations that preserve semantic content."""
    
    def __init__(self):
        self.geometric_transforms = [
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.horizontal_flip,
            self.vertical_flip,
            self.transpose,
        ]
    
    @staticmethod
    def rotate_90(image: torch.Tensor) -> torch.Tensor:
        return TF.rotate(image, 90)
    
    @staticmethod
    def rotate_180(image: torch.Tensor) -> torch.Tensor:
        return TF.rotate(image, 180)
    
    @staticmethod
    def rotate_270(image: torch.Tensor) -> torch.Tensor:
        return TF.rotate(image, 270)
    
    @staticmethod
    def horizontal_flip(image: torch.Tensor) -> torch.Tensor:
        return TF.hflip(image)
    
    @staticmethod
    def vertical_flip(image: torch.Tensor) -> torch.Tensor:
        return TF.vflip(image)
    
    @staticmethod
    def transpose(image: torch.Tensor) -> torch.Tensor:
        # Transpose by rotating 90 degrees and flipping
        return TF.hflip(TF.rotate(image, 90))
    
    def apply_all_transforms(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Apply all geometric transforms."""
        transformed_images = [image]  # Include original
        
        for transform in self.geometric_transforms:
            transformed_images.append(transform(image))
        
        return transformed_images
    
    def apply_random_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Apply a single random geometric transform."""
        transform = random.choice(self.geometric_transforms)
        return transform(image)


class ConsistencyAugmentations:
    """Augmentations designed to test model consistency."""
    
    def __init__(self, strength: float = 0.5):
        self.strength = strength
        
        # Weak augmentations
        self.weak_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
        ])
        
        # Strong augmentations
        self.strong_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
    
    def get_augmented_pair(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a pair of augmented images for consistency training."""
        weak_aug = self.weak_transforms(image)
        strong_aug = self.strong_transforms(image)
        
        return weak_aug, strong_aug


class AdaptiveAugmentation:
    """Augmentation that adapts based on model uncertainty."""
    
    def __init__(self, uncertainty_threshold: float = 0.5):
        self.uncertainty_threshold = uncertainty_threshold
        self.weak_aug = ConsistencyAugmentations(strength=0.3)
        self.strong_aug = ConsistencyAugmentations(strength=0.8)
    
    def __call__(self, image: torch.Tensor, uncertainty: float) -> torch.Tensor:
        """Apply augmentation based on uncertainty level."""
        if uncertainty > self.uncertainty_threshold:
            # High uncertainty - use stronger augmentations
            _, augmented = self.strong_aug.get_augmented_pair(image)
        else:
            # Low uncertainty - use weaker augmentations
            _, augmented = self.weak_aug.get_augmented_pair(image)
        
        return augmented


def create_tta_transforms(num_views: int = 8) -> List[transforms.Compose]:
    """Create a set of test-time augmentation transforms."""
    tta_transforms = []
    
    # Base transform (no augmentation)
    base_transform = transforms.Compose([])
    tta_transforms.append(base_transform)
    
    # Horizontal flip
    if num_views >= 2:
        tta_transforms.append(transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)
        ]))
    
    # Small rotations
    if num_views >= 4:
        for angle in [90, 180, 270]:
            if len(tta_transforms) < num_views:
                tta_transforms.append(transforms.Compose([
                    transforms.Lambda(lambda x: TF.rotate(x, angle))
                ]))
    
    # Color jittering
    if num_views >= 6:
        tta_transforms.append(transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]))
    
    # Scaling
    if num_views >= 8:
        tta_transforms.append(transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.95, 1.05))
        ]))
    
    return tta_transforms[:num_views]