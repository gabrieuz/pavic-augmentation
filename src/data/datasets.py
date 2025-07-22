import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from datasets import load_dataset
import requests
import zipfile
from tqdm import tqdm

from ..config import config

class BaseImageDataset(Dataset):
    def __init__(self, data_path: str, transform=None, target_transform=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        distribution = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

class AdImageNetDataset(BaseImageDataset):
    def __init__(self, split: str = "train", transform=None, target_transform=None):
        super().__init__(
            data_path=os.path.join(config.data_root, "raw", "ad-imagenet"),
            transform=transform,
            target_transform=target_transform
        )
        self.split = split
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        """Download and prepare AdImageNet dataset using Polars"""
        print(f"Loading AdImageNet dataset for {self.split} split...")
        
        try:
            import polars as pl
            from io import BytesIO
            import base64
            
            # Load dataset from Hugging Face
            if self.split == "train":
                df = pl.read_parquet('hf://datasets/PeterBrendan/AdImageNet/data/train-*.parquet')
            else:
                # For validation/test, we'll use a subset of train data
                df = pl.read_parquet('hf://datasets/PeterBrendan/AdImageNet/data/train-*.parquet')
                
                # Split data for val/test
                total_samples = len(df)
                if self.split == "val":
                    df = df.slice(int(total_samples * 0.8), int(total_samples * 0.1))
                elif self.split == "test":
                    df = df.slice(int(total_samples * 0.9), int(total_samples * 0.1))
            
            print(f"Loaded {len(df)} samples from AdImageNet")
            
            # Create output directory
            processed_dir = os.path.join(self.data_path, "processed", self.split)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Get unique labels and create class mapping
            if 'label' in df.columns:
                unique_labels = df['label'].unique().to_list()
            elif 'category' in df.columns:
                unique_labels = df['category'].unique().to_list()
            else:
                # Check all column names
                print(f"Available columns: {df.columns}")
                # Assume the label column might have a different name
                label_candidates = ['label', 'category', 'class', 'target', 'y']
                label_col = None
                for col in label_candidates:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col:
                    unique_labels = df[label_col].unique().to_list()
                else:
                    print("Warning: No label column found, creating dummy labels")
                    unique_labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]
                    label_col = None
            
            self.classes = sorted([str(label) for label in unique_labels])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            print(f"Found {len(self.classes)} classes: {self.classes[:10]}...")
            
            # Process images
            for idx, row in enumerate(tqdm(df.iter_rows(named=True), total=len(df), desc=f"Processing {self.split}")):
                try:
                    # Get image data
                    if 'image' in row:
                        image_data = row['image']
                    elif 'img' in row:
                        image_data = row['img']
                    else:
                        print(f"Warning: No image column found in row {idx}")
                        continue
                    
                    # Get label
                    if label_col and label_col in row:
                        label = str(row[label_col])
                    else:
                        # Assign dummy label
                        label = self.classes[idx % len(self.classes)]
                    
                    if label not in self.class_to_idx:
                        print(f"Warning: Unknown label {label}, skipping...")
                        continue
                    
                    class_idx = self.class_to_idx[label]
                    
                    # Save image
                    img_filename = f"{self.split}_{idx:06d}_class_{class_idx}.jpg"
                    img_path = os.path.join(processed_dir, img_filename)
                    
                    if not os.path.exists(img_path):
                        # Handle different image formats
                        if isinstance(image_data, dict) and 'bytes' in image_data:
                            # Image stored as bytes
                            img_bytes = image_data['bytes']
                            image = Image.open(BytesIO(img_bytes))
                        elif isinstance(image_data, str):
                            # Image stored as base64 string
                            img_bytes = base64.b64decode(image_data)
                            image = Image.open(BytesIO(img_bytes))
                        elif hasattr(image_data, 'save'):
                            # PIL Image object
                            image = image_data
                        else:
                            print(f"Unknown image format for sample {idx}: {type(image_data)}")
                            continue
                        
                        # Convert to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Resize to reasonable size
                        image = image.resize((224, 224), Image.Resampling.LANCZOS)
                        image.save(img_path, 'JPEG', quality=95)
                    
                    self.samples.append((img_path, class_idx))
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
            
            print(f"Successfully processed {len(self.samples)} AdImageNet samples")
            print(f"Classes: {len(self.classes)}")
            
        except ImportError:
            print("Error: polars library not found. Install with: pip install polars")
            print("Creating fallback dummy data...")
            self._create_dummy_data()
        except Exception as e:
            print(f"Error loading AdImageNet dataset: {e}")
            print("Creating fallback dummy data...")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data if real dataset fails to load"""
        self.classes = ["advertisement", "product", "logo", "text", "background"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        processed_dir = os.path.join(self.data_path, "processed", self.split)
        os.makedirs(processed_dir, exist_ok=True)
        
        samples_per_class = 20 if self.split == "train" else 10
        
        for class_idx, class_name in enumerate(self.classes):
            for i in range(samples_per_class):
                from PIL import Image, ImageDraw
                import random
                
                colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                         (255, 255, 100), (255, 100, 255)]
                base_color = colors[class_idx % len(colors)]
                
                img = Image.new('RGB', (224, 224), color=base_color)
                draw = ImageDraw.Draw(img)
                
                for _ in range(random.randint(3, 8)):
                    x = random.randint(0, 200)
                    y = random.randint(0, 200)
                    r = random.randint(10, 30)
                    color = tuple(random.randint(0, 255) for _ in range(3))
                    draw.ellipse([x, y, x+r, y+r], fill=color)
                
                img_filename = f"dummy_{self.split}_{class_name}_{i:03d}.jpg"
                img_path = os.path.join(processed_dir, img_filename)
                img.save(img_path)
                self.samples.append((img_path, class_idx))
        
        print(f"Created {len(self.samples)} dummy AdImageNet samples")

class CIFAR_FS_Dataset(BaseImageDataset):
    def __init__(self, split: str = "train", transform=None, target_transform=None):
        super().__init__(
            data_path=os.path.join(config.data_root, "raw", "cifar-fs"),
            transform=transform,
            target_transform=target_transform
        )
        self.split = split
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        """Download and prepare CIFAR-FS dataset (FC100 format)"""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
            
        # Use FC100 naming convention
        pkl_file = os.path.join(self.data_path, f"FC100_{self.split}.pickle")
        
        if not os.path.exists(pkl_file):
            print(f"FC100 dataset file not found: {pkl_file}")
            print("Please download FC100 dataset files:")
            print("- FC100_train.pickle")
            print("- FC100_val.pickle") 
            print("- FC100_test.pickle")
            return
        
        try:
            # Try different encodings for older pickle files
            data = None
            for encoding in [None, 'latin-1', 'bytes']:
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f, encoding=encoding) if encoding else pickle.load(f)
                    print(f"Successfully loaded with encoding: {encoding or 'default'}")
                    break
                except (UnicodeDecodeError, ValueError) as e:
                    print(f"Failed with encoding {encoding}: {e}")
                    continue
            
            if data is None:
                raise ValueError("Could not load pickle file with any encoding")
            
            # Handle different possible data structures
            if isinstance(data, dict):
                if 'data' in data and 'labels' in data:
                    images = data['data']
                    labels = data['labels']
                elif 'image_data' in data and 'class_dict' in data:
                    # Alternative structure
                    images = data['image_data']
                    labels = data['class_dict']
                else:
                    # Try to extract from any available keys
                    possible_image_keys = ['data', 'images', 'image_data', 'x']
                    possible_label_keys = ['labels', 'targets', 'class_dict', 'y']
                    
                    images = None
                    labels = None
                    
                    for key in possible_image_keys:
                        if key in data:
                            images = data[key]
                            break
                    
                    for key in possible_label_keys:
                        if key in data:
                            labels = data[key]
                            break
                    
                    if images is None or labels is None:
                        print(f"Could not find image/label data in {pkl_file}")
                        print(f"Available keys: {list(data.keys())}")
                        return
            else:
                print(f"Unexpected data format in {pkl_file}")
                return
            
            # Convert labels to list if needed
            if not isinstance(labels, (list, np.ndarray)):
                if hasattr(labels, 'values'):
                    labels = list(labels.values())
                else:
                    labels = list(labels)
            
            # Ensure images is numpy array
            if not isinstance(images, np.ndarray):
                images = np.array(images)
            
            # Create class mapping
            unique_labels = sorted(list(set(labels)))
            self.classes = [f"class_{i}" for i in unique_labels]
            self.class_to_idx = {f"class_{label}": idx for idx, label in enumerate(unique_labels)}
            
            # Process images and labels
            processed_dir = os.path.join(self.data_path, "processed", self.split)
            os.makedirs(processed_dir, exist_ok=True)
            
            print(f"Processing {len(images)} images for {self.split} split...")
            
            for idx, (img, label) in enumerate(zip(images, labels)):
                # Ensure image is in correct format
                if isinstance(img, np.ndarray):
                    if img.max() <= 1.0:  # Normalize if needed
                        img = (img * 255).astype(np.uint8)
                    
                    # Handle different image formats
                    if len(img.shape) == 3:
                        if img.shape[0] == 3:  # CHW format
                            img = np.transpose(img, (1, 2, 0))  # Convert to HWC
                        elif img.shape[2] != 3:  # Not RGB
                            continue
                    else:
                        continue  # Skip invalid images
                    
                    # Create image file
                    img_filename = f"{self.split}_{idx:06d}_class_{label}.png"
                    img_path = os.path.join(processed_dir, img_filename)
                    
                    if not os.path.exists(img_path):
                        try:
                            Image.fromarray(img).save(img_path)
                        except Exception as e:
                            print(f"Error saving image {idx}: {e}")
                            continue
                    
                    # Map to internal class index
                    class_name = f"class_{label}"
                    if class_name in self.class_to_idx:
                        internal_class_idx = self.class_to_idx[class_name]
                        self.samples.append((img_path, internal_class_idx))
            
            print(f"Successfully loaded {len(self.samples)} samples from {pkl_file}")
            print(f"Number of classes: {len(self.classes)}")
            
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            print("Please check if the file exists and has the correct format")

class PlantPathologyDataset(BaseImageDataset):
    def __init__(self, split: str = "train", transform=None, target_transform=None):
        super().__init__(
            data_path=os.path.join(config.data_root, "raw", "plant-pathology"),
            transform=transform,
            target_transform=target_transform
        )
        self.split = split
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        """Create a simple plant pathology dataset"""
        self.classes = ["healthy", "diseased", "pest_damage", "nutrient_deficiency"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        processed_dir = os.path.join(self.data_path, "processed", self.split)
        os.makedirs(processed_dir, exist_ok=True)
        
        samples_per_class = 10 if self.split == "train" else 5
        
        for class_idx, class_name in enumerate(self.classes):
            for i in range(samples_per_class):
                from PIL import Image, ImageDraw
                import random
                
                colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (128, 0, 128)]
                base_color = colors[class_idx % len(colors)]
                
                img = Image.new('RGB', (224, 224), color=base_color)
                draw = ImageDraw.Draw(img)
                
                for _ in range(5):
                    x = random.randint(0, 200)
                    y = random.randint(0, 200)
                    r = random.randint(5, 20)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.ellipse([x, y, x+r, y+r], fill=color)
                
                img_filename = f"{self.split}_{class_name}_{i:03d}.jpg"
                img_path = os.path.join(processed_dir, img_filename)
                img.save(img_path)
                self.samples.append((img_path, class_idx))
        
        print(f"Created {len(self.samples)} plant pathology samples")

def get_default_transforms(image_size: int = 224, is_training: bool = True):
    """Get default image transformations"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataset(dataset_name: str, split: str = "train", transform=None) -> BaseImageDataset:
    """Factory function to get dataset by name"""
    if transform is None:
        transform = get_default_transforms(config.image_size, is_training=(split == "train"))
    
    if dataset_name == "ad-imagenet":
        return AdImageNetDataset(split=split, transform=transform)
    elif dataset_name == "cifar-fs":
        return CIFAR_FS_Dataset(split=split, transform=transform)
    elif dataset_name == "plant-pathology":
        return PlantPathologyDataset(split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_dataloader(dataset: Dataset, batch_size: int = None, shuffle: bool = True) -> DataLoader:
    """Create DataLoader with default settings"""
    if batch_size is None:
        batch_size = config.batch_size
    
    # Adjust settings based on device
    num_workers = config.num_workers if config.device == "cuda" else 0
    pin_memory = config.device == "cuda"
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )