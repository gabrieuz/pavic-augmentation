# PAVIC Augmentation Research Pipeline

## Overview

Research pipeline for evaluating synthetic data generation via fine-tuned diffusion models on class-imbalanced image classification. Compares ResNet and Vision Transformer (ViT) performance across three scenarios:

1. **Imbalanced datasets** (real data only)
2. **Traditional oversampling** (duplicated minority samples)
3. **Synthetic augmentation** (real + diffusion-generated minority samples)

## Research Pipeline

9-stage experimental pipeline:

1. **Dataset Preparation**: AdImageNet, CIFAR-FS, and plant pathology datasets
2. **Imbalance Induction**: Systematic minority class sample removal
3. **Diffusion Fine-tuning**: Fine-tune models on minority samples
4. **Synthetic Generation**: Generate minority samples to match majority counts
5. **Dataset Construction**: Create three dataset versions
6. **Classifier Training**: Train ResNet and ViT on each version
7. **Evaluation**: Comprehensive metrics and confusion matrices
8. **Analysis**: Performance comparison and quality assessment
9. **Documentation**: Reproducible results with detailed logging

## Installation

### Requirements

-   Python 3.8+, GPU (recommended, auto-detects CPU fallback), 8GB+ RAM, 20GB+ disk space

### Setup

```bash
git clone <repository-url>
cd pavic-augmentation
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python main.py
```

### Options

```bash
# Specific datasets
python main.py --datasets ad-imagenet cifar-fs

# Skip stages for testing
python main.py --skip-diffusion
python main.py --skip-training --skip-evaluation

# Custom experiment
python main.py --experiment-name "my_experiment" --seed 123
```

### Command Line Options

-   `--experiment-name`: Custom experiment name
-   `--datasets`: Choose datasets (`ad-imagenet`, `cifar-fs`, `plant-pathology`)
-   `--skip-diffusion`: Skip diffusion fine-tuning and synthetic generation
-   `--skip-training`: Skip classifier training
-   `--skip-evaluation`: Skip model evaluation
-   `--seed`: Random seed (default: 42)

## Configuration

Edit `src/config.py` for key parameters:

-   `datasets`: Dataset selection
-   `minority_classes`: Classes to make minority
-   `imbalance_ratio`: Fraction of minority samples to keep
-   `diffusion_model`: Base diffusion model
-   `num_epochs`: Training epochs
-   `learning_rate`: Learning rate

## Datasets

1. **AdImageNet**: 9,003 advertisement images (auto-downloaded)
2. **CIFAR-FS**: Place FC100 pickle files in `data/raw/cifar-fs/`
3. **Plant Pathology**: Generated locally for testing

## Results

Generated in `results/experiment_name/`:

-   Performance metrics (accuracy, F1, confusion matrices)
-   Comparison visualizations and tables
-   Class distribution analysis
-   Synthetic image quality assessment

## Hardware

-   **RTX 4070**: Automatically detected and optimized (12GB VRAM)
-   **Other GPUs**: 8GB+ VRAM recommended (auto-detects)
-   **CPU**: Automatically adjusts batch size and epochs

## GPU Setup

```bash
# Quick check if your RTX 4070 is detected
python quick_gpu_check.py

# Full GPU setup (if needed)
python check_gpu.py
```

## Troubleshooting

-   **GPU not detected**: Run `python check_gpu.py` to install CUDA PyTorch
-   **Import errors**: `pip install --upgrade diffusers transformers accelerate`
-   **Memory issues**: Automatically optimized per hardware
-   **Testing**: Use `--skip-diffusion` for faster runs
