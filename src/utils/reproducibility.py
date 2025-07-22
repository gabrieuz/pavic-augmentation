import random
import numpy as np
import torch
import os
from typing import Optional

from ..config import config


def set_seed(seed: Optional[int] = None):
    """Define sementes aleatórias para reprodutibilidade"""
    if seed is None:
        seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Garante comportamento determinístico
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define variável de ambiente para hash seed do Python
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to: {seed}")


def setup_device() -> str:
    """Configura e verifica disponibilidade do dispositivo"""
    try:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    except Exception as e:
        device = "cpu"
        print(f"CUDA check failed ({e}), using CPU")

    return device


def print_system_info():
    """Imprime informações do sistema e ambiente"""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {setup_device()}")
    print(f"Random seed: {config.seed}")
    print("="*60)
