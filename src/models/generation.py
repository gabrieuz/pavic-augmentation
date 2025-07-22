import os
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import json

from ..config import config
from .diffusion import DiffusionFineTuner


class SyntheticImageGenerator:
    """Gera imagens sintéticas usando modelos de difusão fine-tuned"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.fine_tuner = DiffusionFineTuner()
        self.pipeline = self.fine_tuner.create_pipeline(checkpoint_path)

    def generate_class_samples(
        self,
        class_name: str,
        class_idx: int,
        num_samples: int,
        output_dir: str,
        guidance_scale: float = None,
        num_inference_steps: int = None,
        seed: int = None
    ) -> List[str]:
        """
        Gera amostras sintéticas para uma classe específica

        Args:
            class_name: Nome da classe para gerar
            class_idx: Índice da classe
            num_samples: Número de amostras a serem geradas
            output_dir: Diretório para salvar as imagens geradas
            guidance_scale: Escala de orientação para a geração
            num_inference_steps: Número de passos de inferência
            seed: Semente aleatória para a geração

        Returns:
            Lista de caminhos para as imagens geradas
        """
        if guidance_scale is None:
            guidance_scale = config.guidance_scale
        if num_inference_steps is None:
            num_inference_steps = config.diffusion_steps
        if seed is None:
            seed = config.seed

        # Define a semente para reprodutibilidade
        torch.manual_seed(seed)
        np.random.seed(seed)

        os.makedirs(output_dir, exist_ok=True)
        generated_paths = []

        # Cria o prompt
        prompt = f"a photo of {class_name}"

        print(f"Generating {num_samples} samples for class '{class_name}'")

        for i in tqdm(range(num_samples), desc=f"Generating {class_name}"):
            # Gera a imagem
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(
                        device=self.pipeline.device).manual_seed(seed + i)
                )

                image = result.images[0]

            # Salva a imagem
            filename = f"{class_name}_synthetic_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            generated_paths.append(filepath)

        # Salva os metadados da geração
        metadata = {
            'class_name': class_name,
            'class_idx': class_idx,
            'num_samples': num_samples,
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'seed': seed,
            'checkpoint_path': self.checkpoint_path,
            'generated_files': [os.path.basename(p) for p in generated_paths]
        }

        metadata_path = os.path.join(
            output_dir, f"{class_name}_generation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return generated_paths

    def generate_minority_samples(
        self,
        class_names: List[str],
        minority_classes: List[int],
        target_distribution: Dict[int, int],
        output_dir: str
    ) -> Dict[int, List[str]]:
        """
        Gera amostras sintéticas para classes minoritárias para corresponder à distribuição alvo

        Args:
            class_names: Lista de todos os nomes de classes
            minority_classes: Lista de índices de classes minoritárias
            target_distribution: Número alvo de amostras por classe
            output_dir: Diretório base para salvar as imagens geradas

        Returns:
            Dicionário mapeando índices de classes para listas de caminhos de imagens geradas
        """
        all_generated = {}

        for class_idx in minority_classes:
            if class_idx >= len(class_names):
                print(
                    f"Warning: Class index {class_idx} out of range for class names")
                continue

            class_name = class_names[class_idx]
            target_count = target_distribution.get(class_idx, 0)

            if target_count <= 0:
                print(
                    f"Skipping class {class_name} (index {class_idx}) - target count is {target_count}")
                continue

            class_output_dir = os.path.join(
                output_dir, f"class_{class_idx}_{class_name}")

            generated_paths = self.generate_class_samples(
                class_name=class_name,
                class_idx=class_idx,
                num_samples=target_count,
                output_dir=class_output_dir
            )

            all_generated[class_idx] = generated_paths

            print(
                f"Generated {len(generated_paths)} samples for class '{class_name}' (index {class_idx})")

        # Salva o resumo geral da geração
        summary = {
            'minority_classes': minority_classes,
            'target_distribution': target_distribution,
            'generated_counts': {cls: len(paths) for cls, paths in all_generated.items()},
            'total_generated': sum(len(paths) for paths in all_generated.values()),
            'output_directory': output_dir
        }

        summary_path = os.path.join(output_dir, "generation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return all_generated


def generate_synthetic_datasets(datasets_info: Dict, checkpoint_paths: Dict[str, str]) -> Dict[str, Dict]:
    """
    Gera imagens sintéticas para todos os conjuntos de dados usando seus modelos fine-tuned

    Args:
        datasets_info: Dicionário contendo informações do conjunto de dados
        checkpoint_paths: Dicionário mapeando nomes de conjuntos de dados para caminhos de checkpoint

    Returns:
        Dicionário contendo os resultados da geração sintética
    """
    results = {}

    for dataset_name, info in datasets_info.items():
        if dataset_name not in checkpoint_paths:
            print(
                f"No checkpoint found for {dataset_name}, skipping synthetic generation...")
            continue

        print(f"\nGenerating synthetic images for {dataset_name}")

        # Cria o gerador
        generator = SyntheticImageGenerator(checkpoint_paths[dataset_name])

        # Calcula a distribuição alvo (corresponder à contagem da classe majoritária)
        imbalanced_dist = info['metadata']['new_distribution']
        max_count = max(imbalanced_dist.values())

        # Alvo: elevar as classes minoritárias à contagem da classe majoritária
        target_distribution = {}
        for class_idx in config.minority_classes:
            current_count = imbalanced_dist.get(class_idx, 0)
            needed = max_count - current_count
            if needed > 0:
                target_distribution[class_idx] = needed

        if not target_distribution:
            print(f"No synthetic samples needed for {dataset_name}")
            continue

        # Gera amostras sintéticas
        output_dir = os.path.join(config.data_root, "synthetic", dataset_name)
        generated_samples = generator.generate_minority_samples(
            class_names=info['original'].classes,
            minority_classes=config.minority_classes,
            target_distribution=target_distribution,
            output_dir=output_dir
        )

        results[dataset_name] = {
            'generated_samples': generated_samples,
            'target_distribution': target_distribution,
            'output_dir': output_dir,
            'checkpoint_path': checkpoint_paths[dataset_name]
        }

        print(f"Synthetic generation completed for {dataset_name}")
        print(
            f"Generated {sum(len(paths) for paths in generated_samples.values())} total samples")

    return results
