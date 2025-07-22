import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    seed: int = 42
    device: str = "auto"  # Will be set automatically based on availability
    
    # Data configuration
    data_root: str = "data"
    datasets: List[str] = None
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    
    # Imbalance configuration
    minority_classes: List[int] = None
    imbalance_ratio: float = 0.1  # Fraction of minority samples to keep
    
    # Diffusion model configuration
    diffusion_model: str = "stabilityai/stable-diffusion-2-1"
    diffusion_steps: int = 50
    guidance_scale: float = 7.5
    num_synthetic_samples: int = 1000
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Training configuration
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 500
    save_every: int = 10
    
    # Evaluation configuration
    test_split: float = 0.2
    val_split: float = 0.1
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["ad-imagenet", "cifar-fs", "plant-pathology"]
        if self.minority_classes is None:
            self.minority_classes = [0, 1, 2]  # Default minority classes
        if self.lora_target_modules is None:
            self.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        
        # Auto-detect device if needed (only print once globally)
        if self.device == "auto":
            import torch
            try:
                if torch.cuda.is_available():
                    self.device = "cuda"
                    
                    # Apply optimizations for RTX 4070
                    gpu_name = torch.cuda.get_device_name()
                    if "4070" in gpu_name:
                        if self.batch_size < 32:
                            self.batch_size = 32
                        if self.num_epochs < 50:
                            self.num_epochs = 50
                else:
                    self.device = "cpu"
            except Exception as e:
                self.device = "cpu"
        
        # Adjust parameters for CPU
        if self.device == "cpu":
            # Reduce batch size for CPU training
            if self.batch_size > 16:
                self.batch_size = 8
            # Reduce epochs for faster CPU training
            if self.num_epochs > 50:
                self.num_epochs = 20
            # Reduce workers for CPU
            if self.num_workers > 2:
                self.num_workers = 0
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.data_root}/raw", exist_ok=True)
        os.makedirs(f"{self.data_root}/processed", exist_ok=True)
        os.makedirs(f"{self.data_root}/synthetic", exist_ok=True)

# Global config instance
config = Config()