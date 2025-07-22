import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
from typing import List, Dict, Optional

from ..config import config
from ..data.datasets import BaseImageDataset


class DiffusionDataset(Dataset):
    """Dataset para treinamento de modelo de difusão"""

    def __init__(self, samples: List[tuple], class_names: List[str], tokenizer, size: int = 512):
        self.samples = samples
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_idx = self.samples[idx]

        # Carrega e pré-processa a imagem
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.size, self.size))
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]

        # Cria prompt de texto
        class_name = self.class_names[class_idx]
        prompt = f"a photo of {class_name}"

        # Tokeniza o prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.squeeze(),
            "attention_mask": text_inputs.attention_mask.squeeze()
        }


class DiffusionFineTuner:
    """Ajuste fino de modelos de difusão em dados de classes minoritárias"""

    def __init__(self, model_id: str = None):
        self.model_id = model_id or config.diffusion_model
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Inicializa componentes
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.scheduler = None

        self._load_models()

    def _load_models(self):
        """Load pretrained diffusion model components"""
        print(f"Loading diffusion model: {self.model_id}")

        # Carrega VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", revision=None
        )

        # Carrega tokenizer e codificador de texto
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer", revision=None
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", revision=None
        )

        # Carrega UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", revision=None
        )

        # Carrega scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler", revision=None
        )

        # Configura LoRA para UNet
        self._setup_lora()

        # Congela VAE e codificador de texto
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Move para o dispositivo
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)

    def prepare_minority_data(self, dataset: BaseImageDataset, minority_classes: List[int]) -> Dataset:
        """Prepara dados de classes minoritárias para ajuste fino"""
        minority_samples = []

        for sample in dataset.samples:
            image_path, class_idx = sample
            if class_idx in minority_classes:
                minority_samples.append(sample)

        print(
            f"Selected {len(minority_samples)} minority samples for fine-tuning")

        return DiffusionDataset(
            samples=minority_samples,
            class_names=dataset.classes,
            tokenizer=self.tokenizer,
            size=512
        )

    def _setup_lora(self):
        """Configura LoRA para ajuste eficiente"""
        print("Setting up LoRA for UNet fine-tuning...")

        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none"
        )

        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel()
                               for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(
            f"LoRA setup complete. Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def fine_tune(
        self,
        dataset: Dataset,
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        save_every: int = 5,
        output_dir: str = None
    ):
        """Ajusta finamente o modelo de difusão"""
        if output_dir is None:
            output_dir = os.path.join(
                config.checkpoint_dir, "diffusion_finetuned")
        os.makedirs(output_dir, exist_ok=True)

        # Cria data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )

        # Configura otimizador (apenas parâmetros do LoRA)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.unet.parameters()),
            lr=learning_rate
        )

        # Prepare with accelerator
        self.unet, optimizer, dataloader = self.accelerator.prepare(
            self.unet, optimizer, dataloader
        )

        # Training loop
        global_step = 0
        for epoch in range(num_epochs):
            self.unet.train()
            epoch_loss = 0.0

            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.unet):
                    # Encode images to latent space
                    with torch.no_grad():
                        latents = self.vae.encode(
                            batch["pixel_values"]).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    batch_size = latents.shape[0]

                    # Sample random timesteps
                    timesteps = torch.randint(
                        0, self.scheduler.config.num_train_timesteps,
                        (batch_size,), device=latents.device
                    ).long()

                    # Add noise to latents
                    noisy_latents = self.scheduler.add_noise(
                        latents, noise, timesteps)

                    # Get text embeddings
                    with torch.no_grad():
                        encoder_hidden_states = self.text_encoder(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"]
                        )[0]

                    # Predict noise
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states
                    ).sample

                    # Compute loss
                    if self.scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.scheduler.config.prediction_type == "v_prediction":
                        target = self.scheduler.get_velocity(
                            latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(),
                                      target.float(), reduction="mean")

                    # Backward pass
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    global_step += 1

                    progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(output_dir, epoch + 1, avg_loss)

        # Save final model
        self.save_checkpoint(output_dir, num_epochs, avg_loss, is_final=True)

        return output_dir

    def save_checkpoint(self, output_dir: str, epoch: int, loss: float, is_final: bool = False):
        """Save model checkpoint"""
        if self.accelerator.is_local_main_process:
            checkpoint_dir = os.path.join(
                output_dir, f"checkpoint-{epoch}" if not is_final else "final")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save LoRA weights
            unet = self.accelerator.unwrap_model(self.unet)
            unet.save_pretrained(checkpoint_dir)

            # Save training info
            training_info = {
                "epoch": epoch,
                "loss": loss,
                "model_id": self.model_id,
                "learning_rate": config.learning_rate,
            }

            with open(os.path.join(checkpoint_dir, "training_info.json"), "w") as f:
                json.dump(training_info, f, indent=2)

            print(f"Checkpoint saved to {checkpoint_dir}")

    def load_finetuned_model(self, checkpoint_path: str):
        """Load fine-tuned LoRA model from checkpoint"""
        print(f"Loading fine-tuned LoRA model from {checkpoint_path}")

        # Load base UNet and apply LoRA
        from peft import PeftModel
        try:
            self.unet = PeftModel.from_pretrained(self.unet, checkpoint_path)
        except:
            # Fallback: reload base model and apply LoRA
            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_id, subfolder="unet", revision=None
            )
            self._setup_lora()
            self.unet = PeftModel.from_pretrained(self.unet, checkpoint_path)

        self.unet.to(self.device)

        # Load training info
        info_path = os.path.join(checkpoint_path, "training_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                training_info = json.load(f)
            print(
                f"Loaded LoRA model from epoch {training_info['epoch']} with loss {training_info['loss']:.4f}")

    def create_pipeline(self, checkpoint_path: str = None) -> StableDiffusionPipeline:
        """Create inference pipeline with fine-tuned model"""
        if checkpoint_path:
            self.load_finetuned_model(checkpoint_path)

        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )

        pipeline.to(self.device)
        return pipeline


def fine_tune_diffusion_models(datasets_info: Dict) -> Dict[str, str]:
    """
    Fine-tune diffusion models for each dataset's minority classes

    Args:
        datasets_info: Dictionary containing dataset information

    Returns:
        Dictionary mapping dataset names to checkpoint paths
    """
    fine_tuner = DiffusionFineTuner()
    checkpoint_paths = {}

    for dataset_name, info in datasets_info.items():
        print(f"\nFine-tuning diffusion model for {dataset_name}")

        # Prepare minority data
        minority_dataset = fine_tuner.prepare_minority_data(
            info['original'],
            config.minority_classes
        )

        if len(minority_dataset) == 0:
            print(f"No minority samples found for {dataset_name}, skipping...")
            continue

        # Fine-tune model
        output_dir = fine_tuner.fine_tune(
            minority_dataset,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            save_every=config.save_every
        )

        checkpoint_paths[dataset_name] = os.path.join(output_dir, "final")
        print(f"Fine-tuning completed for {dataset_name}")

    return checkpoint_paths
