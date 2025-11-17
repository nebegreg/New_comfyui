"""
Intégration avec ComfyUI pour la génération d'images de montagne
Supporte aussi Stable Diffusion direct via diffusers
"""

import requests
import json
import io
import base64
from PIL import Image
from typing import Dict, Optional
import time
import random


class ComfyUIIntegration:
    """Intégration avec ComfyUI API"""

    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(random.randint(0, 1000000))

    def generate_workflow(self, prompt: str, negative_prompt: str, width: int, height: int, steps: int, seed: int) -> Dict:
        """Génère un workflow ComfyUI pour la génération d'image"""
        workflow = {
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 7.5,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "mountain",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        return workflow

    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """Envoie un workflow à ComfyUI"""
        try:
            p = {"prompt": workflow, "client_id": self.client_id}
            response = requests.post(f"http://{self.server_address}/prompt", json=p)
            if response.status_code == 200:
                return response.json().get('prompt_id')
            else:
                print(f"Erreur: {response.status_code}")
                return None
        except Exception as e:
            print(f"Erreur de connexion à ComfyUI: {e}")
            return None

    def get_image(self, prompt_id: str, timeout: int = 300) -> Optional[Image.Image]:
        """Récupère l'image générée"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.server_address}/history/{prompt_id}")
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get('outputs', {})
                        for node_id in outputs:
                            if 'images' in outputs[node_id]:
                                image_data = outputs[node_id]['images'][0]
                                filename = image_data['filename']
                                subfolder = image_data.get('subfolder', '')

                                # Télécharger l'image
                                img_url = f"http://{self.server_address}/view?filename={filename}"
                                if subfolder:
                                    img_url += f"&subfolder={subfolder}"

                                img_response = requests.get(img_url)
                                if img_response.status_code == 200:
                                    return Image.open(io.BytesIO(img_response.content))
                time.sleep(2)
            except Exception as e:
                print(f"Erreur lors de la récupération de l'image: {e}")
                time.sleep(2)

        return None


class StableDiffusionDirect:
    """Alternative: utilisation directe de Stable Diffusion via diffusers"""

    def __init__(self):
        self.pipe = None
        self.device = "cuda"

    def load_model(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Charge le modèle Stable Diffusion"""
        try:
            from diffusers import StableDiffusionXLPipeline
            import torch

            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.pipe.to(self.device)
            print("Modèle Stable Diffusion chargé avec succès")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.device = "cpu"
            return False

    def generate_image(self, prompt: str, negative_prompt: str, width: int, height: int,
                      steps: int, seed: int) -> Optional[Image.Image]:
        """Génère une image avec Stable Diffusion"""
        if self.pipe is None:
            if not self.load_model():
                return None

        try:
            import torch
            generator = torch.Generator(device=self.device).manual_seed(seed)

            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator
            ).images[0]

            return image
        except Exception as e:
            print(f"Erreur lors de la génération: {e}")
            return None
