import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.linalg import sqrtm
from tqdm import tqdm

from ..config import config
from ..training.models import create_model


class ModelEvaluator:
    """Avaliação abrangente de modelos de classificação treinados"""

    def __init__(self, device: str = None):
        self.device = device or config.device

    def load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        model_type: str = None,
        architecture: str = None,
        num_classes: int = None
    ) -> torch.nn.Module:
        """Carrega modelo treinado a partir de um checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Obtém parâmetros do modelo do checkpoint se não fornecidos
        if model_type is None:
            model_type = checkpoint['model_type']
        if architecture is None:
            architecture = checkpoint['architecture']
        if num_classes is None:
            num_classes = checkpoint['num_classes']

        # Cria e carrega o modelo
        model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            architecture=architecture,
            pretrained=False  # Estamos carregando pesos treinados
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model, checkpoint

    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        class_names: List[str] = None
    ) -> Dict:
        """
        Avaliação abrangente de um único modelo

        Args:
            model: Modelo PyTorch treinado
            test_loader: DataLoader de teste
            class_names: Lista de nomes das classes para relatório

        Returns:
            Dicionário com as métricas de avaliação
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Converte para arrays numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)

        # Calcula métricas
        metrics = self._calculate_metrics(y_true, y_pred, y_prob, class_names)

        return metrics

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str] = None
    ) -> Dict:
        """Calcula métricas abrangentes de avaliação"""
        num_classes = len(np.unique(y_true))

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(
            y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # Métricas por classe
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)

        # Métricas por classe especificamente para classes minoritárias
        minority_metrics = {}
        for class_idx in config.minority_classes:
            if class_idx < len(class_names):
                minority_metrics[f'minority_class_{class_idx}'] = {
                    'precision': precision[class_idx] if class_idx < len(precision) else 0.0,
                    'recall': recall[class_idx] if class_idx < len(recall) else 0.0,
                    'f1_score': f1_per_class[class_idx] if class_idx < len(f1_per_class) else 0.0,
                    'support': int(support[class_idx]) if class_idx < len(support) else 0
                }

        # Relatório de classificação
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        return {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1_per_class.tolist(),
                'support': support.tolist()
            },
            'minority_class_metrics': minority_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'class_names': class_names,
            'num_samples': len(y_true),
            'num_classes': num_classes
        }

    def create_confusion_matrix_plot(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_path: str = None,
        normalize: bool = True
    ) -> plt.Figure:
        """Create and save confusion matrix plot"""
        if normalize:
            cm = confusion_matrix.astype(
                'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


class FIDCalculator:
    """Calculate Fréchet Inception Distance (FID) for synthetic image quality assessment"""

    def __init__(self, device: str = None):
        self.device = device or config.device
        self.inception_model = None
        self._load_inception_model()

    def _load_inception_model(self):
        """Load pretrained Inception model for feature extraction"""
        import torchvision.models as models
        self.inception_model = models.inception_v3(
            pretrained=True, transform_input=False)
        self.inception_model.fc = torch.nn.Identity()  # Remove final layer
        self.inception_model.to(self.device)
        self.inception_model.eval()

    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images using Inception model"""
        with torch.no_grad():
            # Resize images to 299x299 for Inception
            if images.size(-1) != 299:
                images = F.interpolate(images, size=(
                    299, 299), mode='bilinear', align_corners=False)

            features = self.inception_model(images)
            return features.cpu().numpy()

    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(
        self,
        real_features: np.ndarray,
        synthetic_features: np.ndarray
    ) -> float:
        """Calculate FID score between real and synthetic features"""
        mu1, sigma1 = self.calculate_statistics(real_features)
        mu2, sigma2 = self.calculate_statistics(synthetic_features)

        # Calculate FID
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))

        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def evaluate_synthetic_quality(
        self,
        real_images_path: str,
        synthetic_images_path: str,
        batch_size: int = 32
    ) -> Dict:
        """
        Evaluate quality of synthetic images using FID

        Args:
            real_images_path: Path to real images directory
            synthetic_images_path: Path to synthetic images directory
            batch_size: Batch size for processing

        Returns:
            Dictionary with FID results
        """
        from torch.utils.data import Dataset
        from torchvision import transforms

        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Load images
        real_features = self._extract_features_from_directory(
            real_images_path, transform, batch_size)
        synthetic_features = self._extract_features_from_directory(
            synthetic_images_path, transform, batch_size)

        if len(real_features) == 0 or len(synthetic_features) == 0:
            return {'error': 'No images found in one or both directories'}

        # Calculate FID
        fid_score = self.calculate_fid(real_features, synthetic_features)

        return {
            'fid_score': fid_score,
            'num_real_images': len(real_features),
            'num_synthetic_images': len(synthetic_features)
        }

    def _extract_features_from_directory(
        self,
        image_dir: str,
        transform,
        batch_size: int
    ) -> np.ndarray:
        """Extract features from all images in a directory"""
        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            return np.array([])

        features_list = []

        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []

            for img_file in batch_files:
                img_path = os.path.join(image_dir, img_file)
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = transform(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error loading image {img_file}: {e}")

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                batch_features = self.extract_features(batch_tensor)
                features_list.append(batch_features)

        if features_list:
            return np.concatenate(features_list, axis=0)
        else:
            return np.array([])


def evaluate_all_models(
    all_splits: Dict,
    training_results: Dict,
    save_results: bool = True
) -> Dict:
    """
    Evaluate all trained models comprehensively

    Args:
        all_splits: Dictionary with all dataset splits
        training_results: Results from training all models
        save_results: Whether to save evaluation results

    Returns:
        Dictionary with comprehensive evaluation results
    """
    evaluator = ModelEvaluator()
    fid_calculator = FIDCalculator()

    all_evaluation_results = {}

    for dataset_name, dataset_results in training_results.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING MODELS FOR {dataset_name.upper()}")
        print(f"{'='*60}")

        dataset_eval_results = {}

        for version_name, version_results in dataset_results.items():
            print(f"\nEvaluating {version_name} version...")

            # Get test data loader
            test_dataset = all_splits[dataset_name][version_name]['test']
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers
            )

            version_eval_results = {}

            for model_key, model_results in version_results.items():
                print(f"  Evaluating {model_key}...")

                # Load best model
                best_model_path = model_results['best_model_path']
                model, checkpoint = evaluator.load_model_from_checkpoint(
                    best_model_path)

                # Evaluate model
                eval_metrics = evaluator.evaluate_model(
                    model=model,
                    test_loader=test_loader,
                    class_names=test_dataset.classes
                )

                # Create confusion matrix plot
                if save_results:
                    cm_save_path = os.path.join(
                        model_results['save_dir'],
                        f"confusion_matrix_{version_name}.png"
                    )
                    evaluator.create_confusion_matrix_plot(
                        confusion_matrix=np.array(
                            eval_metrics['confusion_matrix']),
                        class_names=eval_metrics['class_names'],
                        save_path=cm_save_path
                    )
                    plt.close()

                # Add training info to evaluation results
                eval_metrics.update({
                    'model_type': model_results['model_type'],
                    'architecture': model_results['architecture'],
                    'best_val_acc': model_results['best_val_acc'],
                    'training_epochs': model_results['final_epoch'],
                    'model_path': best_model_path
                })

                version_eval_results[model_key] = eval_metrics

                print(f"    ✓ Test Accuracy: {eval_metrics['accuracy']:.4f}")
                print(
                    f"    ✓ Balanced Accuracy: {eval_metrics['balanced_accuracy']:.4f}")
                print(
                    f"    ✓ F1 Score (weighted): {eval_metrics['f1_weighted']:.4f}")

            dataset_eval_results[version_name] = version_eval_results

        all_evaluation_results[dataset_name] = dataset_eval_results

    # Save comprehensive evaluation results
    if save_results:
        eval_results_path = os.path.join(
            config.results_dir, "comprehensive_evaluation_results.json")
        with open(eval_results_path, 'w') as f:
            json.dump(all_evaluation_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETED")
        print(f"Results saved to: {eval_results_path}")
        print(f"{'='*60}")

    return all_evaluation_results
