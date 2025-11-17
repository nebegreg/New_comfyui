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
    """Intégration avec ComfyUI API - CORRIGÉE pour erreur 400"""

    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(random.randint(0, 1000000))
        self.available_checkpoints = []
        self.default_checkpoint = None

    def test_connection(self) -> bool:
        """Teste la connexion à ComfyUI et découvre les modèles disponibles"""
        try:
            # Test simple de connexion
            response = requests.get(f"http://{self.server_address}/system_stats", timeout=5)
            if response.status_code != 200:
                print(f"❌ ComfyUI ne répond pas (status {response.status_code})")
                return False

            # Récupérer la liste des checkpoints disponibles
            try:
                obj_info_response = requests.get(f"http://{self.server_address}/object_info")
                if obj_info_response.status_code == 200:
                    obj_info = obj_info_response.json()
                    if 'CheckpointLoaderSimple' in obj_info:
                        checkpoint_info = obj_info['CheckpointLoaderSimple']
                        if 'input' in checkpoint_info and 'required' in checkpoint_info['input']:
                            ckpt_options = checkpoint_info['input']['required'].get('ckpt_name', [[]])
                            if isinstance(ckpt_options, list) and len(ckpt_options) > 0:
                                self.available_checkpoints = ckpt_options[0]
                                if self.available_checkpoints:
                                    self.default_checkpoint = self.available_checkpoints[0]
                                    print(f"✓ ComfyUI connecté - {len(self.available_checkpoints)} modèles trouvés")
                                    print(f"  Modèle par défaut: {self.default_checkpoint}")
                                    return True
            except Exception as e:
                print(f"⚠ Impossible de lister les modèles: {e}")

            # Fallback: connexion OK mais pas de liste de modèles
            print("✓ ComfyUI connecté (liste de modèles non disponible)")
            self.default_checkpoint = "sd_xl_base_1.0.safetensors"  # Défaut
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Impossible de se connecter à ComfyUI: {e}")
            print(f"   Vérifiez que ComfyUI est lancé sur {self.server_address}")
            return False

    def get_available_checkpoints(self) -> list:
        """Retourne la liste des checkpoints disponibles"""
        if not self.available_checkpoints:
            self.test_connection()
        return self.available_checkpoints

    def generate_workflow(self, prompt: str, negative_prompt: str, width: int, height: int,
                         steps: int, seed: int, checkpoint: Optional[str] = None) -> Dict:
        """
        Génère un workflow ComfyUI pour la génération d'image

        Args:
            checkpoint: Nom du checkpoint (si None, utilise le défaut)
        """
        # Utiliser le checkpoint spécifié ou le défaut
        if checkpoint is None:
            if self.default_checkpoint is None:
                self.test_connection()
            checkpoint = self.default_checkpoint or "sd_xl_base_1.0.safetensors"

        # Vérifier que le checkpoint existe
        if self.available_checkpoints and checkpoint not in self.available_checkpoints:
            print(f"⚠ Checkpoint '{checkpoint}' non trouvé")
            print(f"   Checkpoints disponibles: {', '.join(self.available_checkpoints[:5])}...")
            # Utiliser le premier disponible
            if self.available_checkpoints:
                checkpoint = self.available_checkpoints[0]
                print(f"   Utilisation de: {checkpoint}")

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
                    "ckpt_name": checkpoint
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
        """Envoie un workflow à ComfyUI avec gestion d'erreurs améliorée"""
        try:
            p = {"prompt": workflow, "client_id": self.client_id}
            response = requests.post(f"http://{self.server_address}/prompt", json=p, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if 'prompt_id' in result:
                    return result['prompt_id']
                elif 'error' in result:
                    print(f"❌ Erreur ComfyUI: {result['error']}")
                    if 'node_errors' in result:
                        print(f"   Erreurs de nodes: {result['node_errors']}")
                    return None
                else:
                    print(f"⚠ Réponse inattendue: {result}")
                    return None

            elif response.status_code == 400:
                # Erreur 400 - Bad Request
                print(f"❌ Erreur 400 - Bad Request")
                try:
                    error_detail = response.json()
                    print(f"   Détails: {error_detail}")

                    # Analyser l'erreur pour aider au debug
                    if 'error' in error_detail:
                        error_msg = error_detail['error']
                        if 'node_errors' in error_detail:
                            print(f"   Nodes problématiques:")
                            for node_id, node_error in error_detail['node_errors'].items():
                                print(f"     - Node {node_id}: {node_error}")

                        # Suggestions courantes
                        if 'ckpt_name' in str(error_detail).lower():
                            print(f"\n   💡 Suggestion: Le checkpoint spécifié n'existe pas")
                            print(f"      Checkpoints disponibles: {self.available_checkpoints[:3]}")
                        elif 'required' in str(error_detail).lower():
                            print(f"\n   💡 Suggestion: Il manque des paramètres requis dans le workflow")

                except Exception as parse_err:
                    print(f"   Impossible de parser l'erreur: {parse_err}")
                    print(f"   Réponse brute: {response.text}")

                return None

            else:
                print(f"❌ Erreur HTTP {response.status_code}")
                print(f"   Réponse: {response.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            print(f"❌ Timeout lors de la connexion à ComfyUI")
            print(f"   Vérifiez que ComfyUI est lancé et accessible")
            return None
        except Exception as e:
            print(f"❌ Erreur lors de la communication avec ComfyUI: {e}")
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
