import os
import json
import time
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import wandb

from ..config import config
from ..data.datasets import create_dataloader
from .models import create_model, get_model_info


class ClassificationTrainer:
    """Treinador para modelos de classificação (ResNet e ViT)"""

    def __init__(
        self,
        model_type: str,
        num_classes: int,
        architecture: str = None,
        device: str = None,
        use_wandb: bool = False
    ):
        self.model_type = model_type
        self.num_classes = num_classes
        self.architecture = architecture
        self.device = device or config.device
        self.use_wandb = use_wandb

        # Garante que o dispositivo é válido
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU")
            self.device = "cpu"

        # Inicializa o modelo
        self.model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            architecture=architecture,
            pretrained=True,
            dropout=0.1
        ).to(self.device)

        # Obtém informações do modelo
        self.model_info = get_model_info(self.model)
        print(
            f"Created {model_type} model with {self.model_info['total_parameters']:,} parameters")

        # Componentes de treinamento
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

        # Estado do treinamento
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'epochs': []
        }

    def setup_training(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine"
    ):
        """Configura otimizador e scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Treina por uma época"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Passagem para frente
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Passagem para trás
            loss.backward()
            self.optimizer.step()

            # Estatísticas
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Atualiza barra de progresso
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, Dict]:
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions,
                      average='weighted', zero_division=0)

        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )

        class_metrics = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support.tolist()
        }

        return avg_loss, accuracy, f1, class_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        save_dir: str = None,
        save_every: int = 10,
        early_stopping_patience: int = 20
    ) -> Dict:
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience

        Returns:
            Training results dictionary
        """
        if num_epochs is None:
            num_epochs = config.num_epochs

        if save_dir is None:
            save_dir = os.path.join(
                config.checkpoint_dir,
                f"{self.model_type}_{self.architecture}_{int(time.time())}"
            )
        os.makedirs(save_dir, exist_ok=True)

        # Setup training if not already done
        if self.optimizer is None:
            self.setup_training()

        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model: {self.model_type} ({self.architecture})")
        print(f"Device: {self.device}")
        print(f"Checkpoint directory: {save_dir}")

        best_model_path = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_f1, class_metrics = self.validate(
                val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['epochs'].append(epoch + 1)

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Print epoch results
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(
                f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_path = os.path.join(save_dir, "best_model.pth")
                self.save_checkpoint(
                    best_model_path, epoch + 1, val_acc, class_metrics)
                epochs_without_improvement = 0
                print(f"  ✓ New best model saved (Val Acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1

            # Save regular checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(
                    save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                self.save_checkpoint(
                    checkpoint_path, epoch + 1, val_acc, class_metrics)

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
                break

        # Save final training results
        results = {
            'model_type': self.model_type,
            'architecture': self.architecture,
            'model_info': self.model_info,
            'best_val_acc': self.best_val_acc,
            'final_epoch': self.current_epoch + 1,
            'training_history': self.training_history,
            'best_model_path': best_model_path,
            'save_dir': save_dir
        }

        results_path = os.path.join(save_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Results saved to: {results_path}")

        return results

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        val_acc: float,
        class_metrics: Dict
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'model_type': self.model_type,
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'model_info': self.model_info,
            'class_metrics': class_metrics,
            'training_history': self.training_history
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'epochs': []
        })

        print(
            f"Loaded checkpoint from epoch {self.current_epoch} with val_acc: {self.best_val_acc:.4f}")

        return checkpoint


def train_all_models(
    all_splits: Dict,
    model_configs: List[Dict] = None
) -> Dict:
    """
    Train all model configurations on all dataset versions

    Args:
        all_splits: Dictionary with all dataset splits
        model_configs: List of model configurations to train

    Returns:
        Dictionary with all training results
    """
    if model_configs is None:
        model_configs = [
            {'model_type': 'resnet', 'architecture': 'resnet50'},
            {'model_type': 'vit', 'architecture': 'vit_base_patch16_224'}
        ]

    all_results = {}

    for dataset_name, dataset_splits in all_splits.items():
        print(f"\n{'='*60}")
        print(f"TRAINING MODELS FOR {dataset_name.upper()}")
        print(f"{'='*60}")

        dataset_results = {}

        for version_name, splits in dataset_splits.items():
            print(f"\nDataset version: {version_name}")

            # Get number of classes
            num_classes = len(splits['train'].classes)

            # Create data loaders
            train_loader = create_dataloader(splits['train'], shuffle=True)
            val_loader = create_dataloader(splits['val'], shuffle=False)

            version_results = {}

            for model_config in model_configs:
                model_key = f"{model_config['model_type']}_{model_config['architecture']}"
                print(f"\nTraining {model_key}...")

                # Create trainer
                trainer = ClassificationTrainer(
                    model_type=model_config['model_type'],
                    num_classes=num_classes,
                    architecture=model_config['architecture'],
                    device=config.device
                )

                # Train model
                results = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=config.num_epochs,
                    save_every=config.save_every
                )

                version_results[model_key] = results

                print(f"✓ {model_key} training completed")
                print(f"  Best Val Acc: {results['best_val_acc']:.4f}")

            dataset_results[version_name] = version_results

        all_results[dataset_name] = dataset_results

    # Save comprehensive results
    results_path = os.path.join(
        config.results_dir, "comprehensive_training_results.json")
    with open(results_path, 'w') as f:
        # Convert to JSON-serializable format
        serializable_results = {}
        for dataset_name, dataset_results in all_results.items():
            serializable_results[dataset_name] = {}
            for version_name, version_results in dataset_results.items():
                serializable_results[dataset_name][version_name] = {}
                for model_key, results in version_results.items():
                    # Keep only JSON-serializable parts
                    serializable_results[dataset_name][version_name][model_key] = {
                        'model_type': results['model_type'],
                        'architecture': results['architecture'],
                        'best_val_acc': results['best_val_acc'],
                        'final_epoch': results['final_epoch'],
                        'model_info': results['model_info'],
                        'save_dir': results['save_dir']
                    }

        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL TRAINING COMPLETED")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")

    return all_results
