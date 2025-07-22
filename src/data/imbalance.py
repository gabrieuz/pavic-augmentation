import os
import json
import random
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter

from ..config import config
from .datasets import BaseImageDataset

class ImbalanceInducer:
    """Class to systematically induce imbalance in datasets"""
    
    def __init__(self, seed: int = None):
        self.seed = seed or config.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def induce_imbalance(
        self, 
        dataset: BaseImageDataset,
        minority_classes: List[int],
        imbalance_ratio: float = 0.1,
        save_indices: bool = True
    ) -> Tuple[BaseImageDataset, Dict]:
        """
        Induce imbalance by removing samples from minority classes
        
        Args:
            dataset: The original balanced dataset
            minority_classes: List of class indices to make minority
            imbalance_ratio: Fraction of minority samples to keep
            save_indices: Whether to save removed indices for reproducibility
            
        Returns:
            Tuple of (imbalanced_dataset, metadata)
        """
        original_distribution = dataset.get_class_distribution()
        removed_indices = {cls: [] for cls in minority_classes}
        
        # Group samples by class
        class_samples = {cls: [] for cls in original_distribution.keys()}
        for idx, (_, label) in enumerate(dataset.samples):
            class_samples[label].append(idx)
        
        # Calculate target counts for minority classes
        target_counts = {}
        for cls in minority_classes:
            if cls in class_samples:
                original_count = len(class_samples[cls])
                target_count = int(original_count * imbalance_ratio)
                target_counts[cls] = target_count
            else:
                print(f"Warning: Class {cls} not found in dataset, skipping...")
                target_counts[cls] = 0
        
        # Remove samples from minority classes
        indices_to_remove = set()
        for cls in minority_classes:
            if cls in class_samples and class_samples[cls]:
                samples_indices = class_samples[cls].copy()
                random.shuffle(samples_indices)
                
                target_count = target_counts[cls]
                to_remove = samples_indices[target_count:]
                
                removed_indices[cls] = to_remove
                indices_to_remove.update(to_remove)
            else:
                removed_indices[cls] = []
        
        # Create new dataset with remaining samples
        new_samples = []
        for idx, sample in enumerate(dataset.samples):
            if idx not in indices_to_remove:
                new_samples.append(sample)
        
        # Create imbalanced dataset
        imbalanced_dataset = BaseImageDataset(dataset.data_path, dataset.transform, dataset.target_transform)
        imbalanced_dataset.samples = new_samples
        imbalanced_dataset.classes = dataset.classes
        imbalanced_dataset.class_to_idx = dataset.class_to_idx
        
        # Create metadata
        new_distribution = imbalanced_dataset.get_class_distribution()
        metadata = {
            'seed': self.seed,
            'minority_classes': minority_classes,
            'imbalance_ratio': imbalance_ratio,
            'original_distribution': original_distribution,
            'new_distribution': new_distribution,
            'removed_indices': removed_indices,
            'total_removed': len(indices_to_remove),
            'target_counts': target_counts
        }
        
        # Save metadata for reproducibility
        if save_indices:
            self._save_metadata(metadata, dataset.__class__.__name__)
        
        return imbalanced_dataset, metadata
    
    def _save_metadata(self, metadata: Dict, dataset_name: str):
        """Save imbalance metadata to file"""
        metadata_dir = os.path.join(config.data_root, "processed", "imbalance_metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        filename = f"{dataset_name}_imbalance_seed_{self.seed}.json"
        filepath = os.path.join(metadata_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Imbalance metadata saved to: {filepath}")
    
    def load_metadata(self, dataset_name: str, seed: int = None) -> Dict:
        """Load previously saved imbalance metadata"""
        if seed is None:
            seed = self.seed
            
        metadata_dir = os.path.join(config.data_root, "processed", "imbalance_metadata")
        filename = f"{dataset_name}_imbalance_seed_{seed}.json"
        filepath = os.path.join(metadata_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    def apply_saved_imbalance(self, dataset: BaseImageDataset, metadata: Dict) -> BaseImageDataset:
        """Apply previously saved imbalance configuration"""
        indices_to_remove = set()
        for cls_indices in metadata['removed_indices'].values():
            indices_to_remove.update(cls_indices)
        
        new_samples = []
        for idx, sample in enumerate(dataset.samples):
            if idx not in indices_to_remove:
                new_samples.append(sample)
        
        imbalanced_dataset = BaseImageDataset(dataset.data_path, dataset.transform, dataset.target_transform)
        imbalanced_dataset.samples = new_samples
        imbalanced_dataset.classes = dataset.classes
        imbalanced_dataset.class_to_idx = dataset.class_to_idx
        
        return imbalanced_dataset

class OversamplingBalancer:
    """Class to balance datasets using traditional oversampling"""
    
    def __init__(self, seed: int = None):
        self.seed = seed or config.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def balance_by_oversampling(self, dataset: BaseImageDataset) -> BaseImageDataset:
        """
        Balance dataset by duplicating minority class samples
        
        Args:
            dataset: The imbalanced dataset to balance
            
        Returns:
            Balanced dataset with oversampled minority classes
        """
        distribution = dataset.get_class_distribution()
        max_count = max(distribution.values())
        
        # Group samples by class
        class_samples = {cls: [] for cls in distribution.keys()}
        for idx, (path, label) in enumerate(dataset.samples):
            class_samples[label].append((path, label))
        
        # Oversample minority classes
        balanced_samples = []
        for cls, samples in class_samples.items():
            current_count = len(samples)
            target_count = max_count
            
            # Add original samples
            balanced_samples.extend(samples)
            
            # Add duplicated samples to reach target count
            if current_count < target_count:
                needed = target_count - current_count
                duplicates = random.choices(samples, k=needed)
                balanced_samples.extend(duplicates)
        
        # Shuffle the balanced samples
        random.shuffle(balanced_samples)
        
        # Create balanced dataset
        balanced_dataset = BaseImageDataset(dataset.data_path, dataset.transform, dataset.target_transform)
        balanced_dataset.samples = balanced_samples
        balanced_dataset.classes = dataset.classes
        balanced_dataset.class_to_idx = dataset.class_to_idx
        
        return balanced_dataset

def create_imbalanced_datasets(datasets: List[BaseImageDataset]) -> Dict[str, Dict]:
    """
    Create imbalanced versions of all datasets
    
    Args:
        datasets: List of original balanced datasets
        
    Returns:
        Dictionary with imbalanced datasets and metadata
    """
    inducer = ImbalanceInducer()
    results = {}
    
    # Filter out empty datasets
    valid_datasets = []
    for dataset in datasets:
        if len(dataset.samples) > 0:
            valid_datasets.append(dataset)
        else:
            print(f"Warning: Skipping empty dataset {dataset.__class__.__name__}")
    
    if not valid_datasets:
        raise RuntimeError("No valid datasets found. All datasets are empty.")
    
    for dataset in valid_datasets:
        dataset_name = dataset.__class__.__name__
        
        # Check if dataset has the required minority classes
        available_classes = set(label for _, label in dataset.samples)
        missing_classes = set(config.minority_classes) - available_classes
        
        if missing_classes:
            print(f"Warning: Dataset {dataset_name} missing minority classes: {missing_classes}")
            # Adjust minority classes to only include available ones
            valid_minority_classes = [cls for cls in config.minority_classes if cls in available_classes]
        else:
            valid_minority_classes = config.minority_classes
        
        if not valid_minority_classes:
            print(f"Warning: No valid minority classes for {dataset_name}, using first available class")
            valid_minority_classes = [min(available_classes)] if available_classes else []
        
        if valid_minority_classes:
            # Create imbalanced version
            imbalanced_dataset, metadata = inducer.induce_imbalance(
                dataset,
                minority_classes=valid_minority_classes,
                imbalance_ratio=config.imbalance_ratio
            )
            
            # Create oversampled version
            balancer = OversamplingBalancer()
            oversampled_dataset = balancer.balance_by_oversampling(imbalanced_dataset)
            
            results[dataset_name] = {
                'original': dataset,
                'imbalanced': imbalanced_dataset,
                'oversampled': oversampled_dataset,
                'metadata': metadata
            }
            
            print(f"Created imbalanced datasets for {dataset_name}")
            print(f"Original distribution: {metadata['original_distribution']}")
            print(f"Imbalanced distribution: {metadata['new_distribution']}")
            print(f"Oversampled distribution: {oversampled_dataset.get_class_distribution()}")
            print("-" * 50)
        else:
            print(f"Skipping {dataset_name} - no valid classes found")
    
    if not results:
        raise RuntimeError("No datasets could be processed successfully.")
    
    return results