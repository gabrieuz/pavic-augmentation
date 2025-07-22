import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont
import random

from ..config import config


class ResultsAnalyzer:
    """Análise e visualização abrangente dos resultados experimentais"""

    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir or config.results_dir
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_performance_comparison_table(
        self,
        evaluation_results: Dict,
        save_path: str = None
    ) -> pd.DataFrame:
        """Cria uma tabela de comparação de desempenho abrangente"""

        rows = []

        for dataset_name, dataset_results in evaluation_results.items():
            for version_name, version_results in dataset_results.items():
                for model_key, metrics in version_results.items():

                    # Extrai as métricas das classes minoritárias
                    minority_f1_scores = []
                    minority_recalls = []
                    minority_precisions = []

                    for minority_metrics in metrics['minority_class_metrics'].values():
                        minority_f1_scores.append(minority_metrics['f1_score'])
                        minority_recalls.append(minority_metrics['recall'])
                        minority_precisions.append(
                            minority_metrics['precision'])

                    avg_minority_f1 = np.mean(
                        minority_f1_scores) if minority_f1_scores else 0
                    avg_minority_recall = np.mean(
                        minority_recalls) if minority_recalls else 0
                    avg_minority_precision = np.mean(
                        minority_precisions) if minority_precisions else 0

                    row = {
                        'Dataset': dataset_name,
                        'Version': version_name,
                        'Model': model_key,
                        'Architecture': metrics['architecture'],
                        'Accuracy': metrics['accuracy'],
                        'Balanced_Accuracy': metrics['balanced_accuracy'],
                        'F1_Weighted': metrics['f1_weighted'],
                        'F1_Macro': metrics['f1_macro'],
                        'Avg_Minority_F1': avg_minority_f1,
                        'Avg_Minority_Recall': avg_minority_recall,
                        'Avg_Minority_Precision': avg_minority_precision,
                        'Training_Epochs': metrics['training_epochs'],
                        'Best_Val_Acc': metrics['best_val_acc']
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Tabela de desempenho salva em: {save_path}")

        return df

    def plot_performance_comparison(
        self,
        df: pd.DataFrame,
        save_path: str = None
    ) -> plt.Figure:
        """Cria gráficos de comparação de desempenho abrangentes"""

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            'Comparação de Desempenho do Modelo Entre Versões do Dataset', fontsize=16)

        metrics_to_plot = [
            ('Accuracy', 'Acurácia Geral'),
            ('Balanced_Accuracy', 'Acurácia Balanceada'),
            ('F1_Weighted', 'F1 Score (Ponderado)'),
            ('F1_Macro', 'F1 Score (Macro)'),
            ('Avg_Minority_F1', 'F1 Médio Minoritário'),
            ('Avg_Minority_Recall', 'Recall Médio Minoritário')
        ]

        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]

            # Cria tabela dinâmica para o heatmap
            pivot_data = df.pivot_table(
                values=metric,
                index=['Dataset', 'Version'],
                columns='Model',
                aggfunc='mean'
            )

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'shrink': 0.8}
            )
            ax.set_title(title)
            ax.set_xlabel('')
            ax.set_ylabel('')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de comparação de desempenho salvo em: {save_path}")

        return fig

    def create_version_comparison_charts(
        self,
        df: pd.DataFrame,
        save_dir: str = None
    ) -> Dict[str, plt.Figure]:
        """Cria gráficos de comparação detalhados para diferentes versões do dataset"""

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        figures = {}

        # 1. Gráfico de barras comparando versões para cada dataset
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Análise de Desempenho - {dataset}', fontsize=14)

            metrics = ['Accuracy', 'Balanced_Accuracy',
                       'F1_Weighted', 'Avg_Minority_F1']
            metric_titles = ['Acurácia', 'Acurácia Balanceada',
                             'F1 Ponderado', 'F1 Médio Minoritário']

            for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
                ax = axes[idx // 2, idx % 2]

                sns.barplot(
                    data=dataset_df,
                    x='Version',
                    y=metric,
                    hue='Model',
                    ax=ax
                )
                ax.set_title(title)
                ax.set_ylabel('Pontuação')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()

            if save_dir:
                save_path = os.path.join(
                    save_dir, f'{dataset}_version_comparison.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            figures[f'{dataset}_version_comparison'] = fig

        # 2. Comparação de modelos entre datasets
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            'Desempenho do Modelo Entre Datasets e Versões', fontsize=14)

        # Comparação ResNet vs ViT
        for idx, metric in enumerate(['Accuracy', 'Avg_Minority_F1']):
            ax = axes[idx]

            sns.boxplot(
                data=df,
                x='Model',
                y=metric,
                hue='Version',
                ax=ax
            )
            ax.set_title(f'Distribuição de {metric}')
            ax.set_ylabel('Pontuação')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'model_comparison_boxplot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        figures['model_comparison'] = fig

        return figures

    def create_synthetic_quality_analysis(
        self,
        fid_results: Dict,
        save_path: str = None
    ) -> plt.Figure:
        """Analisa e visualiza a qualidade da imagem sintética usando pontuações FID"""

        if not fid_results:
            print("Nenhum resultado de FID disponível para análise")
            return None

        # Prepara os dados para plotagem
        datasets = []
        fid_scores = []
        num_synthetic = []

        for dataset_name, results in fid_results.items():
            if 'fid_score' in results:
                datasets.append(dataset_name)
                fid_scores.append(results['fid_score'])
                num_synthetic.append(results.get('num_synthetic_images', 0))

        if not datasets:
            print("Nenhuma pontuação FID válida encontrada")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Análise de Qualidade da Imagem Sintética', fontsize=14)

        # Pontuações FID por dataset
        bars1 = ax1.bar(datasets, fid_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Pontuações FID por Dataset')
        ax1.set_ylabel('Pontuação FID (Menor é Melhor)')
        ax1.set_xlabel('Dataset')

        # Adiciona rótulos de valor nas barras
        for bar, score in zip(bars1, fid_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                     f'{score:.2f}', ha='center', va='bottom')

        # Número de imagens sintéticas geradas
        bars2 = ax2.bar(datasets, num_synthetic, color='lightcoral', alpha=0.7)
        ax2.set_title('Número de Imagens Sintéticas Geradas')
        ax2.set_ylabel('Contagem')
        ax2.set_xlabel('Dataset')

        # Adiciona rótulos de valor nas barras
        for bar, count in zip(bars2, num_synthetic):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                     f'{count}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de análise FID salvo em: {save_path}")

        return fig

    def create_class_distribution_analysis(
        self,
        datasets_info: Dict,
        save_path: str = None
    ) -> plt.Figure:
        """Analisa e visualiza as distribuições de classes entre as versões do dataset"""

        fig, axes = plt.subplots(
            len(datasets_info), 3, figsize=(18, 6*len(datasets_info)))
        if len(datasets_info) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            'Análise de Distribuição de Classes Entre Versões do Dataset', fontsize=16)

        version_names = ['imbalanced', 'oversampled', 'synthetic_augmented']
        version_titles = ['Desbalanceado',
                          'Oversampled', 'Aumentado Sinteticamente']

        for dataset_idx, (dataset_name, info) in enumerate(datasets_info.items()):
            for version_idx, (version_name, version_title) in enumerate(zip(version_names, version_titles)):
                ax = axes[dataset_idx, version_idx] if len(
                    datasets_info) > 1 else axes[version_idx]

                if version_name in info:
                    distribution = info[version_name].get_class_distribution()
                    classes = list(distribution.keys())
                    counts = list(distribution.values())

                    # Cria mapa de cores para classes minoritárias vs majoritárias
                    colors = [
                        'red' if cls in config.minority_classes else 'blue' for cls in classes]

                    bars = ax.bar(range(len(classes)), counts,
                                  color=colors, alpha=0.7)
                    ax.set_title(f'{dataset_name} - {version_title}')
                    ax.set_xlabel('Índice da Classe')
                    ax.set_ylabel('Número de Amostras')
                    ax.set_xticks(range(len(classes)))
                    ax.set_xticklabels(classes)

                    # Adiciona rótulos de valor nas barras
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{count}', ha='center', va='bottom')
                else:
                    ax.text(0.5, 0.5, 'Dados Não Disponíveis', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{dataset_name} - {version_title}')

        # Adiciona legenda
        red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7)
        blue_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7)
        fig.legend([red_patch, blue_patch], ['Classes Minoritárias', 'Classes Majoritárias'],
                   loc='upper right', bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Análise de distribuição de classes salva em: {save_path}")

        return fig

    def create_synthetic_samples_grid(
        self,
        synthetic_results: Dict,
        samples_per_class: int = 4,
        save_path: str = None
    ) -> Optional[plt.Figure]:
        """Cria uma grade mostrando exemplos de amostras sintéticas"""

        if not synthetic_results:
            print("Nenhum resultado sintético disponível")
            return None

        # Coleta imagens de amostra
        sample_images = {}

        for dataset_name, results in synthetic_results.items():
            if 'generated_samples' not in results:
                continue

            dataset_samples = {}
            for class_idx, image_paths in results['generated_samples'].items():
                if image_paths:
                    # Amostra imagens aleatoriamente
                    selected_paths = random.sample(
                        image_paths,
                        min(samples_per_class, len(image_paths))
                    )
                    dataset_samples[class_idx] = selected_paths

            if dataset_samples:
                sample_images[dataset_name] = dataset_samples

        if not sample_images:
            print("Nenhuma imagem de amostra encontrada")
            return None

        # Cria a figura
        num_datasets = len(sample_images)
        max_classes = max(len(classes) for classes in sample_images.values())

        fig, axes = plt.subplots(
            num_datasets * max_classes,
            samples_per_class,
            figsize=(samples_per_class * 3, num_datasets * max_classes * 3)
        )

        if num_datasets == 1 and max_classes == 1:
            axes = axes.reshape(1, -1)
        elif num_datasets == 1 or max_classes == 1:
            axes = axes.reshape(-1, samples_per_class)

        fig.suptitle(
            'Amostras de Imagens Sintéticas por Dataset e Classe', fontsize=16)

        row_idx = 0
        for dataset_name, dataset_samples in sample_images.items():
            for class_idx, image_paths in dataset_samples.items():
                for col_idx in range(samples_per_class):
                    if num_datasets == 1 and max_classes == 1:
                        ax = axes[col_idx]
                    else:
                        ax = axes[row_idx, col_idx]

                    if col_idx < len(image_paths):
                        try:
                            image = Image.open(image_paths[col_idx])
                            ax.imshow(image)
                            ax.axis('off')
                            if col_idx == 0:
                                ax.set_ylabel(f'{dataset_name}\nClasse {class_idx}', rotation=0,
                                              ha='right', va='center')
                        except Exception as e:
                            ax.text(0.5, 0.5, 'Erro\nAo Carregar\nA Imagem',
                                    ha='center', va='center')
                            ax.axis('off')
                    else:
                        ax.axis('off')

                row_idx += 1

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grade de amostras sintéticas salva em: {save_path}")

        return fig

    def generate_comprehensive_report(
        self,
        evaluation_results: Dict,
        datasets_info: Dict = None,
        synthetic_results: Dict = None,
        fid_results: Dict = None,
        save_dir: str = None
    ) -> str:
        """Gera um relatório de análise abrangente com todas as visualizações"""

        if save_dir is None:
            save_dir = os.path.join(self.results_dir, "comprehensive_analysis")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Gerando relatório de análise abrangente em: {save_dir}")

        # 1. Tabela de comparação de desempenho
        print("Criando tabela de comparação de desempenho...")
        df = self.create_performance_comparison_table(
            evaluation_results,
            save_path=os.path.join(save_dir, "performance_table.csv")
        )

        # 2. Gráficos de comparação de desempenho
        print("Criando gráficos de comparação de desempenho...")
        self.plot_performance_comparison(
            df,
            save_path=os.path.join(save_dir, "performance_comparison.png")
        )
        plt.close()

        # 3. Gráficos de comparação de versão
        print("Criando gráficos de comparação de versão...")
        version_figs = self.create_version_comparison_charts(
            df,
            save_dir=os.path.join(save_dir, "version_comparisons")
        )
        for fig in version_figs.values():
            plt.close(fig)

        # 4. Análise de distribuição de classes
        if datasets_info:
            print("Criando análise de distribuição de classes...")
            self.create_class_distribution_analysis(
                datasets_info,
                save_path=os.path.join(save_dir, "class_distributions.png")
            )
            plt.close()

        # 5. Análise FID
        if fid_results:
            print("Criando análise FID...")
            self.create_synthetic_quality_analysis(
                fid_results,
                save_path=os.path.join(save_dir, "fid_analysis.png")
            )
            plt.close()

        # 6. Grade de amostras sintéticas
        if synthetic_results:
            print("Criando visualização de amostras sintéticas...")
            self.create_synthetic_samples_grid(
                synthetic_results,
                save_path=os.path.join(save_dir, "synthetic_samples.png")
            )
            plt.close()

        # 7. Gera estatísticas de resumo
        summary_stats = self._generate_summary_statistics(df)
        with open(os.path.join(save_dir, "summary_statistics.json"), 'w') as f:
            json.dump(summary_stats, f, indent=2)

        print(f"✓ Relatório de análise abrangente gerado em: {save_dir}")
        return save_dir

    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Gera estatísticas de resumo a partir dos resultados da avaliação"""

        summary = {
            'overall_statistics': {
                'total_experiments': len(df),
                'datasets_tested': df['Dataset'].nunique(),
                'models_tested': df['Model'].nunique(),
                'versions_tested': df['Version'].nunique()
            },
            'best_performers': {
                'highest_accuracy': {
                    'value': df['Accuracy'].max(),
                    'experiment': df.loc[df['Accuracy'].idxmax()].to_dict()
                },
                'highest_minority_f1': {
                    'value': df['Avg_Minority_F1'].max(),
                    'experiment': df.loc[df['Avg_Minority_F1'].idxmax()].to_dict()
                }
            },
            'version_comparison': {
                'avg_accuracy_by_version': df.groupby('Version')['Accuracy'].mean().to_dict(),
                'avg_minority_f1_by_version': df.groupby('Version')['Avg_Minority_F1'].mean().to_dict()
            },
            'model_comparison': {
                'avg_accuracy_by_model': df.groupby('Model')['Accuracy'].mean().to_dict(),
                'avg_minority_f1_by_model': df.groupby('Model')['Avg_Minority_F1'].mean().to_dict()
            }
        }

        return summary
