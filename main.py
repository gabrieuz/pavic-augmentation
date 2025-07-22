#!/usr/bin/env python3
"""
Executor principal de experimentos para o Pipeline de Pesquisa de Aumento PAVIC

Este script orquestra o pipeline experimental completo para avaliar
o impacto da geração de dados sintéticos via modelos de difusão ajustados
na classificação de imagens com classes desbalanceadas.

Pipeline de Pesquisa:
1. Preparação do Dataset (Mini-ImageNet, CIFAR-FS, Plant Pathology)
2. Indução de Desbalanceamento nas classes minoritárias
3. Ajuste Fino do Modelo de Difusão em amostras minoritárias
4. Geração de Imagens Sintéticas
5. Construção do Dataset (3 versões: desbalanceado, sobreamostrado, aumentado-sintético)
6. Treinamento de Classificadores (ResNet e ViT)
7. Avaliação e Análise Abrangente
"""

from src.visualization.analysis import ResultsAnalyzer
from src.evaluation.metrics import evaluate_all_models, FIDCalculator
from src.training.trainer import train_all_models
from src.models.generation import generate_synthetic_datasets
from src.models.diffusion import fine_tune_diffusion_models
from src.data.construction import construct_final_datasets
from src.data.imbalance import create_imbalanced_datasets
from src.data.datasets import get_dataset
from src.utils.reproducibility import set_seed, setup_device, print_system_info
from src.config import config
import os
import sys
import argparse
import json
import time
from typing import Dict, List

# Adiciona src ao caminho
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class ExperimentRunner:
    """Orquestrador principal do experimento"""

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"pavic_experiment_{int(time.time())}"
        self.start_time = time.time()

        # Setup experiment directory
        self.experiment_dir = os.path.join(
            config.results_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        print(f"Starting experiment: {self.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")

        # Initialize components
        self.datasets_info = {}
        self.synthetic_results = {}
        self.training_results = {}
        self.evaluation_results = {}
        self.fid_results = {}

    def run_complete_pipeline(
        self,
        datasets: List[str] = None,
        skip_diffusion: bool = False,
        skip_training: bool = False,
        skip_evaluation: bool = False
    ):
        """Executa o pipeline experimental completo"""

        print_system_info()
        set_seed(config.seed)

        try:
            # Estágio 1: Preparação do Dataset e Indução de Desbalanceamento
            print("\n" + "="*60)
            print("ESTÁGIO 1: PREPARAÇÃO DO DATASET E INDUÇÃO DE DESBALANCEAMENTO")
            print("="*60)
            self._prepare_datasets(datasets)

            # Estágio 2: Ajuste Fino do Modelo de Difusão (Opcional)
            if not skip_diffusion:
                print("\n" + "="*60)
                print("ESTÁGIO 2: AJUSTE FINO DO MODELO DE DIFUSÃO")
                print("="*60)
                self._fine_tune_diffusion_models()

                # Estágio 3: Geração de Imagens Sintéticas
                print("\n" + "="*60)
                print("ESTÁGIO 3: GERAÇÃO DE IMAGENS SINTÉTICAS")
                print("="*60)
                self._generate_synthetic_images()

                # Estágio 4: Avaliação FID
                print("\n" + "="*60)
                print("ESTÁGIO 4: AVALIAÇÃO DE QUALIDADE DAS IMAGENS SINTÉTICAS")
                print("="*60)
                self._evaluate_synthetic_quality()
            else:
                print("\nPulando o ajuste fino do modelo de difusão e geração sintética")

            # Estágio 5: Construção do Dataset
            print("\n" + "="*60)
            print("ESTÁGIO 5: CONSTRUÇÃO DO DATASET")
            print("="*60)
            self._construct_datasets()

            # Estágio 6: Treinamento do Classificador (Opcional)
            if not skip_training:
                print("\n" + "="*60)
                print("ESTÁGIO 6: TREINAMENTO DO CLASSIFICADOR")
                print("="*60)
                self._train_classifiers()
            else:
                print("\nPulando o treinamento do classificador")

            # Estágio 7: Avaliação do Modelo (Opcional)
            if not skip_evaluation and not skip_training:
                print("\n" + "="*60)
                print("ESTÁGIO 7: AVALIAÇÃO DO MODELO")
                print("="*60)
                self._evaluate_models()
            else:
                print("\nPulando a avaliação do modelo")

            # Estágio 8: Análise e Visualização
            print("\n" + "="*60)
            print("ESTÁGIO 8: ANÁLISE E VISUALIZAÇÃO")
            print("="*60)
            self._generate_analysis()

            # Save experiment summary
            self._save_experiment_summary()

            print("\n" + "="*60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(
                f"Total runtime: {(time.time() - self.start_time) / 3600:.2f} hours")
            print(f"Results saved in: {self.experiment_dir}")
            print("="*60)

        except Exception as e:
            print(f"\nERROR: Experiment failed with exception: {e}")
            raise e

    def _prepare_datasets(self, datasets: List[str] = None):
        """Prepara os datasets e induz desbalanceamento"""
        if datasets is None:
            datasets = config.datasets

        print(f"Preparing datasets: {datasets}")

        # Load original datasets
        original_datasets = []
        for dataset_name in datasets:
            try:
                dataset = get_dataset(dataset_name, split="train")
                original_datasets.append(dataset)
                print(
                    f"✓ Loaded {dataset_name}: {len(dataset.samples)} samples, {len(dataset.classes)} classes")
            except Exception as e:
                print(f"✗ Failed to load {dataset_name}: {e}")

        if not original_datasets:
            raise RuntimeError("No datasets could be loaded successfully")

        # Create imbalanced versions
        print("\nCreating imbalanced dataset versions...")
        self.datasets_info = create_imbalanced_datasets(original_datasets)

        print(f"✓ Successfully prepared {len(self.datasets_info)} datasets")

    def _fine_tune_diffusion_models(self):
        """Realiza ajuste fino dos modelos de difusão nas classes minoritárias"""
        print("Fine-tuning diffusion models...")

        checkpoint_paths = fine_tune_diffusion_models(self.datasets_info)

        # Save checkpoint paths
        checkpoint_info_path = os.path.join(
            self.experiment_dir, "diffusion_checkpoints.json")
        with open(checkpoint_info_path, 'w') as f:
            json.dump(checkpoint_paths, f, indent=2)

        print(
            f"✓ Fine-tuning completed. Checkpoints saved: {checkpoint_info_path}")
        return checkpoint_paths

    def _generate_synthetic_images(self):
        """Gera imagens sintéticas usando modelos com ajuste fino"""
        print("Generating synthetic images...")

        # Load checkpoint paths
        checkpoint_info_path = os.path.join(
            self.experiment_dir, "diffusion_checkpoints.json")
        if os.path.exists(checkpoint_info_path):
            with open(checkpoint_info_path, 'r') as f:
                checkpoint_paths = json.load(f)
        else:
            print("No checkpoint paths found, skipping synthetic generation")
            return {}

        self.synthetic_results = generate_synthetic_datasets(
            self.datasets_info, checkpoint_paths)

        # Save synthetic results
        synthetic_info_path = os.path.join(
            self.experiment_dir, "synthetic_generation_info.json")
        with open(synthetic_info_path, 'w') as f:
            # Create JSON-serializable version
            serializable_results = {}
            for dataset_name, results in self.synthetic_results.items():
                serializable_results[dataset_name] = {
                    'target_distribution': results['target_distribution'],
                    'output_dir': results['output_dir'],
                    'checkpoint_path': results['checkpoint_path'],
                    'generated_counts': {
                        str(cls): len(paths) for cls, paths in results['generated_samples'].items()
                    }
                }
            json.dump(serializable_results, f, indent=2)

        print(
            f"✓ Synthetic generation completed. Info saved: {synthetic_info_path}")

    def _evaluate_synthetic_quality(self):
        """Avalia a qualidade das imagens sintéticas usando FID"""
        print("Evaluating synthetic image quality...")

        fid_calculator = FIDCalculator()
        self.fid_results = {}

        for dataset_name, results in self.synthetic_results.items():
            if 'generated_samples' not in results:
                continue

            print(f"Calculating FID for {dataset_name}...")

            # For each minority class, compare real vs synthetic
            dataset_fid_results = {}

            for class_idx, synthetic_paths in results['generated_samples'].items():
                if not synthetic_paths:
                    continue

                # Get real images for this class
                real_samples = [
                    path for path, label in self.datasets_info[dataset_name]['original'].samples
                    if label == class_idx
                ]

                if not real_samples:
                    continue

                # Create temporary directories for comparison
                real_dir = os.path.join(
                    results['output_dir'], f"temp_real_class_{class_idx}")
                synthetic_dir = os.path.join(
                    results['output_dir'], f"class_{class_idx}_*")

                # Find the actual synthetic directory
                import glob
                synthetic_dirs = glob.glob(synthetic_dir)
                if not synthetic_dirs:
                    continue

                synthetic_dir = synthetic_dirs[0]

                try:
                    # Create real images directory
                    os.makedirs(real_dir, exist_ok=True)

                    # Copy some real images for comparison
                    import random
                    import shutil
                    selected_real = random.sample(real_samples, min(
                        len(real_samples), len(synthetic_paths)))

                    for i, real_path in enumerate(selected_real):
                        dest_path = os.path.join(real_dir, f"real_{i}.jpg")
                        shutil.copy2(real_path, dest_path)

                    # Calculate FID
                    fid_result = fid_calculator.evaluate_synthetic_quality(
                        real_images_path=real_dir,
                        synthetic_images_path=synthetic_dir
                    )

                    dataset_fid_results[f'class_{class_idx}'] = fid_result

                    # Clean up temporary directory
                    shutil.rmtree(real_dir, ignore_errors=True)

                except Exception as e:
                    print(
                        f"Error calculating FID for {dataset_name} class {class_idx}: {e}")

            self.fid_results[dataset_name] = dataset_fid_results

        # Save FID results
        fid_results_path = os.path.join(
            self.experiment_dir, "fid_results.json")
        with open(fid_results_path, 'w') as f:
            json.dump(self.fid_results, f, indent=2)

        print(f"✓ FID evaluation completed. Results saved: {fid_results_path}")

    def _construct_datasets(self):
        """Constrói os datasets finais com todas as três versões"""
        print("Constructing final dataset versions...")

        self.all_versions, self.all_splits = construct_final_datasets(
            self.datasets_info,
            self.synthetic_results
        )

        print("✓ Dataset construction completed")

    def _train_classifiers(self):
        """Treina os classificadores ResNet e ViT"""
        print("Training classification models...")

        self.training_results = train_all_models(self.all_splits)

        print("✓ Classifier training completed")

    def _evaluate_models(self):
        """Avalia os modelos treinados de forma abrangente"""
        print("Evaluating trained models...")

        self.evaluation_results = evaluate_all_models(
            self.all_splits,
            self.training_results
        )

        print("✓ Model evaluation completed")

    def _generate_analysis(self):
        """Gera análises abrangentes e visualizações"""
        print("Generating analysis and visualizations...")

        analyzer = ResultsAnalyzer(results_dir=self.experiment_dir)

        analysis_dir = analyzer.generate_comprehensive_report(
            evaluation_results=self.evaluation_results,
            datasets_info=self.datasets_info,
            synthetic_results=self.synthetic_results,
            fid_results=self.fid_results,
            save_dir=os.path.join(self.experiment_dir, "analysis")
        )

        print(f"✓ Analysis completed. Report saved in: {analysis_dir}")

    def _save_experiment_summary(self):
        """Salva o resumo completo do experimento"""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration_hours': (time.time() - self.start_time) / 3600,
            'configuration': {
                'seed': config.seed,
                'datasets': config.datasets,
                'minority_classes': config.minority_classes,
                'imbalance_ratio': config.imbalance_ratio,
                'diffusion_model': config.diffusion_model,
                'num_epochs': config.num_epochs,
                'image_size': config.image_size
            },
            'results_summary': {
                'datasets_processed': len(self.datasets_info),
                'synthetic_datasets_generated': len(self.synthetic_results),
                'models_trained': len(self.training_results) if hasattr(self, 'training_results') else 0,
                'models_evaluated': len(self.evaluation_results) if hasattr(self, 'evaluation_results') else 0
            },
            'experiment_directory': self.experiment_dir
        }

        summary_path = os.path.join(
            self.experiment_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Experiment summary saved: {summary_path}")


def main():
    """Ponto de entrada principal"""
    parser = argparse.ArgumentParser(
        description="PAVIC Augmentation Research Pipeline")

    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name for this experiment'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['ad-imagenet', 'cifar-fs', 'plant-pathology'],
        help='Datasets to use (default: all)'
    )
    parser.add_argument(
        '--skip-diffusion',
        action='store_true',
        help='Skip diffusion model fine-tuning and synthetic generation'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip classifier training'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip model evaluation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config default)'
    )

    args = parser.parse_args()

    # Update config with command line arguments
    if args.seed:
        config.seed = args.seed
    if args.epochs:
        config.num_epochs = args.epochs

    # Run experiment
    runner = ExperimentRunner(experiment_name=args.experiment_name)

    runner.run_complete_pipeline(
        datasets=args.datasets,
        skip_diffusion=args.skip_diffusion,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )


if __name__ == "__main__":
    main()
