import os
import json
import shutil
from typing import Dict, List, Tuple
from collections import defaultdict

from ..config import config
from .datasets import BaseImageDataset
from .imbalance import ImbalanceInducer, OversamplingBalancer

class DatasetConstructor:
    """Construct the three dataset versions: imbalanced, oversampled, and synthetic-augmented"""
    
    def __init__(self):
        self.processed_dir = os.path.join(config.data_root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def construct_all_versions(
        self,
        datasets_info: Dict,
        synthetic_results: Dict
    ) -> Dict[str, Dict[str, BaseImageDataset]]:
        """
        Construct all three dataset versions for each dataset
        
        Args:
            datasets_info: Dictionary containing original and imbalanced datasets
            synthetic_results: Dictionary containing synthetic generation results
            
        Returns:
            Dictionary with all dataset versions
        """
        all_versions = {}
        
        for dataset_name, info in datasets_info.items():
            print(f"\nConstructing dataset versions for {dataset_name}")
            
            versions = {}
            
            # Version 1: Imbalanced (already available)
            versions['imbalanced'] = info['imbalanced']
            print(f"✓ Imbalanced version: {len(info['imbalanced'].samples)} samples")
            
            # Version 2: Oversampled (already available)
            versions['oversampled'] = info['oversampled']
            print(f"✓ Oversampled version: {len(info['oversampled'].samples)} samples")
            
            # Version 3: Synthetic-augmented
            if dataset_name in synthetic_results:
                synthetic_augmented = self._create_synthetic_augmented_dataset(
                    imbalanced_dataset=info['imbalanced'],
                    synthetic_results=synthetic_results[dataset_name],
                    dataset_name=dataset_name
                )
                versions['synthetic_augmented'] = synthetic_augmented
                print(f"✓ Synthetic-augmented version: {len(synthetic_augmented.samples)} samples")
            else:
                print(f"⚠️  No synthetic results available for {dataset_name}")
                versions['synthetic_augmented'] = info['imbalanced']  # Fallback to imbalanced
            
            all_versions[dataset_name] = versions
            
            # Save version information
            self._save_version_info(dataset_name, versions)
        
        return all_versions
    
    def _create_synthetic_augmented_dataset(
        self,
        imbalanced_dataset: BaseImageDataset,
        synthetic_results: Dict,
        dataset_name: str
    ) -> BaseImageDataset:
        """
        Create synthetic-augmented dataset by combining real and synthetic samples
        
        Args:
            imbalanced_dataset: The original imbalanced dataset
            synthetic_results: Results from synthetic generation
            dataset_name: Name of the dataset
            
        Returns:
            Synthetic-augmented dataset
        """
        # Start with real samples from imbalanced dataset
        augmented_samples = imbalanced_dataset.samples.copy()
        
        # Add synthetic samples
        generated_samples = synthetic_results['generated_samples']
        
        for class_idx, image_paths in generated_samples.items():
            for image_path in image_paths:
                augmented_samples.append((image_path, class_idx))
        
        # Create new dataset
        synthetic_dataset = BaseImageDataset(
            data_path=imbalanced_dataset.data_path,
            transform=imbalanced_dataset.transform,
            target_transform=imbalanced_dataset.target_transform
        )
        synthetic_dataset.samples = augmented_samples
        synthetic_dataset.classes = imbalanced_dataset.classes
        synthetic_dataset.class_to_idx = imbalanced_dataset.class_to_idx
        
        return synthetic_dataset
    
    def _save_version_info(self, dataset_name: str, versions: Dict[str, BaseImageDataset]):
        """Save information about dataset versions"""
        version_info = {
            'dataset_name': dataset_name,
            'versions': {}
        }
        
        for version_name, dataset in versions.items():
            distribution = dataset.get_class_distribution()
            version_info['versions'][version_name] = {
                'total_samples': len(dataset.samples),
                'class_distribution': distribution,
                'num_classes': len(distribution),
                'balance_ratio': min(distribution.values()) / max(distribution.values()) if distribution else 0
            }
        
        # Save to file
        info_path = os.path.join(self.processed_dir, f"{dataset_name}_versions_info.json")
        with open(info_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print(f"Version info saved to: {info_path}")
    
    def get_dataset_statistics(self, all_versions: Dict[str, Dict[str, BaseImageDataset]]) -> Dict:
        """Get comprehensive statistics for all dataset versions"""
        statistics = {}
        
        for dataset_name, versions in all_versions.items():
            dataset_stats = {
                'dataset_name': dataset_name,
                'versions': {}
            }
            
            for version_name, dataset in versions.items():
                distribution = dataset.get_class_distribution()
                
                version_stats = {
                    'total_samples': len(dataset.samples),
                    'num_classes': len(distribution),
                    'class_distribution': distribution,
                    'min_class_size': min(distribution.values()) if distribution else 0,
                    'max_class_size': max(distribution.values()) if distribution else 0,
                    'balance_ratio': min(distribution.values()) / max(distribution.values()) if distribution else 0,
                    'minority_classes': [cls for cls, count in distribution.items() 
                                       if count == min(distribution.values())],
                    'majority_classes': [cls for cls, count in distribution.items() 
                                       if count == max(distribution.values())]
                }
                
                dataset_stats['versions'][version_name] = version_stats
            
            statistics[dataset_name] = dataset_stats
        
        # Save comprehensive statistics
        stats_path = os.path.join(self.processed_dir, "comprehensive_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        return statistics
    
    def create_splits(
        self, 
        all_versions: Dict[str, Dict[str, BaseImageDataset]],
        test_split: float = None,
        val_split: float = None,
        seed: int = None
    ) -> Dict[str, Dict[str, Dict[str, BaseImageDataset]]]:
        """
        Create train/validation/test splits for all dataset versions
        
        Args:
            all_versions: All dataset versions
            test_split: Fraction for test set
            val_split: Fraction for validation set
            seed: Random seed for splitting
            
        Returns:
            Dictionary with train/val/test splits for all versions
        """
        if test_split is None:
            test_split = config.test_split
        if val_split is None:
            val_split = config.val_split
        if seed is None:
            seed = config.seed
        
        import random
        random.seed(seed)
        
        all_splits = {}
        
        for dataset_name, versions in all_versions.items():
            dataset_splits = {}
            
            for version_name, dataset in versions.items():
                # Group samples by class for stratified splitting
                class_samples = defaultdict(list)
                for idx, (path, label) in enumerate(dataset.samples):
                    class_samples[label].append((path, label))
                
                train_samples = []
                val_samples = []
                test_samples = []
                
                # Split each class separately to maintain class distribution
                for class_idx, samples in class_samples.items():
                    random.shuffle(samples)
                    
                    n_samples = len(samples)
                    n_test = int(n_samples * test_split)
                    n_val = int(n_samples * val_split)
                    n_train = n_samples - n_test - n_val
                    
                    test_samples.extend(samples[:n_test])
                    val_samples.extend(samples[n_test:n_test + n_val])
                    train_samples.extend(samples[n_test + n_val:])
                
                # Shuffle the splits
                random.shuffle(train_samples)
                random.shuffle(val_samples)
                random.shuffle(test_samples)
                
                # Create split datasets
                splits = {}
                for split_name, split_samples in [
                    ('train', train_samples),
                    ('val', val_samples),
                    ('test', test_samples)
                ]:
                    split_dataset = BaseImageDataset(
                        data_path=dataset.data_path,
                        transform=dataset.transform,
                        target_transform=dataset.target_transform
                    )
                    split_dataset.samples = split_samples
                    split_dataset.classes = dataset.classes
                    split_dataset.class_to_idx = dataset.class_to_idx
                    
                    splits[split_name] = split_dataset
                
                dataset_splits[version_name] = splits
                
                print(f"{dataset_name} - {version_name}: "
                      f"Train: {len(train_samples)}, "
                      f"Val: {len(val_samples)}, "
                      f"Test: {len(test_samples)}")
            
            all_splits[dataset_name] = dataset_splits
        
        # Save split information
        self._save_split_info(all_splits, test_split, val_split, seed)
        
        return all_splits
    
    def _save_split_info(self, all_splits: Dict, test_split: float, val_split: float, seed: int):
        """Save information about dataset splits"""
        split_info = {
            'test_split': test_split,
            'val_split': val_split,
            'seed': seed,
            'datasets': {}
        }
        
        for dataset_name, dataset_splits in all_splits.items():
            dataset_info = {'versions': {}}
            
            for version_name, splits in dataset_splits.items():
                version_info = {}
                for split_name, split_dataset in splits.items():
                    distribution = split_dataset.get_class_distribution()
                    version_info[split_name] = {
                        'total_samples': len(split_dataset.samples),
                        'class_distribution': distribution
                    }
                dataset_info['versions'][version_name] = version_info
            
            split_info['datasets'][dataset_name] = dataset_info
        
        # Save to file
        info_path = os.path.join(self.processed_dir, "split_info.json")
        with open(info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Split info saved to: {info_path}")

def construct_final_datasets(
    datasets_info: Dict,
    synthetic_results: Dict
) -> Tuple[Dict, Dict]:
    """
    Main function to construct all dataset versions and splits
    
    Args:
        datasets_info: Dictionary containing original and imbalanced datasets
        synthetic_results: Dictionary containing synthetic generation results
        
    Returns:
        Tuple of (all_versions, all_splits)
    """
    constructor = DatasetConstructor()
    
    # Construct all versions
    all_versions = constructor.construct_all_versions(datasets_info, synthetic_results)
    
    # Get comprehensive statistics
    statistics = constructor.get_dataset_statistics(all_versions)
    print("\n" + "="*60)
    print("DATASET CONSTRUCTION SUMMARY")
    print("="*60)
    
    for dataset_name, stats in statistics.items():
        print(f"\n{dataset_name.upper()}:")
        for version_name, version_stats in stats['versions'].items():
            print(f"  {version_name}: {version_stats['total_samples']} samples, "
                  f"balance ratio: {version_stats['balance_ratio']:.3f}")
    
    # Create train/val/test splits
    all_splits = constructor.create_splits(all_versions)
    
    print(f"\n✓ All dataset versions and splits created successfully!")
    print(f"✓ Statistics saved to: {constructor.processed_dir}")
    
    return all_versions, all_splits