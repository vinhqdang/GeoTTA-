import torch
import gc
from typing import Dict, List, Any
import psutil
import subprocess


def optimize_memory():
    """Optimize GPU and CPU memory usage."""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Force garbage collection
    gc.collect()


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    cached = torch.cuda.memory_reserved() / 1024**3  # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    free = total - allocated
    
    return {
        'total': total,
        'allocated': allocated, 
        'cached': cached,
        'free': free
    }


def get_cpu_memory_info() -> Dict[str, float]:
    """Get CPU memory information."""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / 1024**3,  # GB
        'available': memory.available / 1024**3,  # GB
        'used': memory.used / 1024**3,  # GB
        'percentage': memory.percent
    }


def print_memory_stats():
    """Print current memory statistics."""
    gpu_info = get_gpu_memory_info()
    cpu_info = get_cpu_memory_info()
    
    print(f"GPU Memory: {gpu_info['allocated']:.2f}GB / {gpu_info['total']:.2f}GB "
          f"({gpu_info['allocated']/gpu_info['total']*100:.1f}%)")
    print(f"CPU Memory: {cpu_info['used']:.2f}GB / {cpu_info['total']:.2f}GB "
          f"({cpu_info['percentage']:.1f}%)")


class MemoryProfiler:
    """Context manager for profiling memory usage."""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.start_gpu = None
        self.start_cpu = None
    
    def __enter__(self):
        self.start_gpu = get_gpu_memory_info()
        self.start_cpu = get_cpu_memory_info()
        if self.description:
            print(f"Starting {self.description}")
            self.print_memory("Start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_gpu = get_gpu_memory_info()
        end_cpu = get_cpu_memory_info()
        
        if self.description:
            print(f"Finished {self.description}")
            self.print_memory("End")
            
            gpu_diff = end_gpu['allocated'] - self.start_gpu['allocated']
            cpu_diff = end_cpu['used'] - self.start_cpu['used']
            
            print(f"Memory change - GPU: {gpu_diff:+.3f}GB, CPU: {cpu_diff:+.3f}GB")
    
    def print_memory(self, stage: str):
        gpu_info = get_gpu_memory_info()
        cpu_info = get_cpu_memory_info()
        print(f"{stage} - GPU: {gpu_info['allocated']:.3f}GB, "
              f"CPU: {cpu_info['used']:.3f}GB")


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        with MemoryProfiler(f"function {func.__name__}"):
            result = func(*args, **kwargs)
        return result
    return wrapper


def check_memory_requirements(model: torch.nn.Module, 
                            input_shape: tuple,
                            batch_size: int = 1) -> Dict[str, float]:
    """
    Estimate memory requirements for model inference.
    
    Args:
        model: PyTorch model
        input_shape: Shape of single input (C, H, W)
        batch_size: Batch size to test
        
    Returns:
        Dictionary with memory estimates
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Measure memory before
    torch.cuda.empty_cache()
    before_memory = torch.cuda.memory_allocated()
    
    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Measure memory after
    after_memory = torch.cuda.memory_allocated()
    
    memory_per_sample = (after_memory - before_memory) / batch_size / 1024**2  # MB
    
    # Clean up
    del dummy_input
    torch.cuda.empty_cache()
    
    return {
        'memory_per_sample_mb': memory_per_sample,
        'memory_per_batch_mb': memory_per_sample * batch_size,
        'max_batch_size_8gb': int(8 * 1024 / memory_per_sample) if memory_per_sample > 0 else 0
    }


def setup_memory_efficient_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup memory-efficient training configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with memory optimizations
    """
    # Check available GPU memory
    gpu_info = get_gpu_memory_info()
    available_memory_gb = gpu_info['free']
    
    print(f"Available GPU memory: {available_memory_gb:.2f}GB")
    
    # Adjust batch size based on available memory
    if available_memory_gb < 6:  # Less than 6GB
        config['training']['batch_size'] = min(config['training']['batch_size'], 8)
        config['training']['grad_accum_steps'] = max(config['training']['grad_accum_steps'], 8)
        config['training']['mixed_precision'] = True
        print("Low memory mode: batch_size=8, grad_accum=8, mixed_precision=True")
    
    elif available_memory_gb < 10:  # Less than 10GB  
        config['training']['batch_size'] = min(config['training']['batch_size'], 16)
        config['training']['grad_accum_steps'] = max(config['training']['grad_accum_steps'], 4)
        config['training']['mixed_precision'] = True
        print("Medium memory mode: batch_size=16, grad_accum=4, mixed_precision=True")
    
    # Enable gradient checkpointing for very low memory
    if available_memory_gb < 4:
        config['training']['gradient_checkpointing'] = True
        print("Enabled gradient checkpointing due to very low memory")
    
    return config


def clear_model_cache(model: torch.nn.Module):
    """Clear any cached computations in the model."""
    # Clear CLIP model cache if it exists
    if hasattr(model, 'clip_model'):
        if hasattr(model.clip_model, 'logit_scale'):
            # Clear any cached features
            torch.cuda.empty_cache()
    
    # Clear text prototype cache
    if hasattr(model, 'text_prototype_cache'):
        model.text_prototype_cache = None
    
    # Force garbage collection
    gc.collect()


def get_model_memory_footprint(model: torch.nn.Module) -> Dict[str, float]:
    """Get memory footprint of model parameters."""
    param_memory = 0
    buffer_memory = 0
    
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    
    total_memory = param_memory + buffer_memory
    
    return {
        'parameters_mb': param_memory / 1024**2,
        'buffers_mb': buffer_memory / 1024**2,
        'total_mb': total_memory / 1024**2,
        'total_gb': total_memory / 1024**3
    }


def suggest_optimal_batch_size(model: torch.nn.Module, 
                              input_shape: tuple,
                              target_memory_usage: float = 0.8) -> int:
    """
    Suggest optimal batch size based on available memory.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (C, H, W)
        target_memory_usage: Target memory usage fraction (0.8 = 80%)
        
    Returns:
        Suggested batch size
    """
    gpu_info = get_gpu_memory_info()
    available_memory = gpu_info['free'] * target_memory_usage * 1024  # MB
    
    # Test with batch size 1
    memory_req = check_memory_requirements(model, input_shape, batch_size=1)
    memory_per_sample = memory_req['memory_per_sample_mb']
    
    if memory_per_sample <= 0:
        return 1
    
    suggested_batch_size = int(available_memory / memory_per_sample)
    return max(1, suggested_batch_size)