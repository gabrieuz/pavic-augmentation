import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional


class ResNetClassifier(nn.Module):
    """Classificador ResNet com arquitetura customizável"""

    def __init__(
        self,
        num_classes: int,
        architecture: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture

        # Carrega ResNet pré-treinado
        if architecture == "resnet18":
            self.backbone = models.resnet18(
                weights='DEFAULT' if pretrained else None)
            feature_dim = 512
        elif architecture == "resnet34":
            self.backbone = models.resnet34(
                weights='DEFAULT' if pretrained else None)
            feature_dim = 512
        elif architecture == "resnet50":
            self.backbone = models.resnet50(
                weights='DEFAULT' if pretrained else None)
            feature_dim = 2048
        elif architecture == "resnet101":
            self.backbone = models.resnet101(
                weights='DEFAULT' if pretrained else None)
            feature_dim = 2048
        else:
            raise ValueError(
                f"Unsupported ResNet architecture: {architecture}")

        # Substitui o classificador final
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        """Extrai features antes da camada de classificação"""
        features = self.backbone.avgpool(self.backbone.layer4(
            self.backbone.layer3(self.backbone.layer2(
                self.backbone.layer1(self.backbone.maxpool(
                    self.backbone.relu(self.backbone.bn1(
                        self.backbone.conv1(x)))))))))
        return features.flatten(1)


class ViTClassifier(nn.Module):
    """Classificador Vision Transformer usando a biblioteca timm"""

    def __init__(
        self,
        num_classes: int,
        architecture: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture

        # Carrega ViT pré-treinado
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove head
            drop_rate=dropout
        )

        # Obtém dimensão dos features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).shape[1]

        # Adiciona head de classificação customizado
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extrai features antes da camada de classificação"""
        return self.backbone(x)


def create_model(
    model_type: str,
    num_classes: int,
    architecture: str = None,
    pretrained: bool = True,
    dropout: float = 0.1
) -> nn.Module:
    """
    Função fábrica para criar modelos de classificação

    Args:
        model_type: 'resnet' ou 'vit'
        num_classes: Número de classes de saída
        architecture: Variante específica da arquitetura
        pretrained: Se deve usar pesos pré-treinados
        dropout: Taxa de dropout

    Returns:
        Modelo PyTorch
    """
    if model_type.lower() == 'resnet':
        if architecture is None:
            architecture = "resnet50"
        return ResNetClassifier(
            num_classes=num_classes,
            architecture=architecture,
            pretrained=pretrained,
            dropout=dropout
        )

    elif model_type.lower() == 'vit':
        if architecture is None:
            architecture = "vit_base_patch16_224"
        return ViTClassifier(
            num_classes=num_classes,
            architecture=architecture,
            pretrained=pretrained,
            dropout=dropout
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Use 'resnet' or 'vit'.")


def get_model_info(model: nn.Module) -> dict:
    """Obtém informações sobre o modelo"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': model.__class__.__name__,
        'architecture': getattr(model, 'architecture', 'unknown')
    }
