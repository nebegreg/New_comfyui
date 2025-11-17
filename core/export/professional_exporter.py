"""
Professional Exporter for Mountain Studio Pro v2.0
Exports all generated assets in industry-standard formats
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)


class ProfessionalExporter:
    """
    Exportateur professionnel pour tous les assets générés

    Formats supportés:
    - Images: PNG (8-bit, 16-bit), EXR (32-bit HDR)
    - 3D: OBJ, FBX (si disponible)
    - Data: JSON (végétation, metadata)
    - Textures: PBR splatmaps, normal maps, AO, roughness
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Dossier de sortie pour tous les exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ProfessionalExporter initialized: {self.output_dir}")

    def export_heightmap(
        self,
        heightmap: np.ndarray,
        filename: str = "heightmap.png",
        bit_depth: int = 16
    ):
        """
        Exporte une heightmap en PNG 8-bit ou 16-bit

        Args:
            heightmap: Heightmap normalisée (0-1)
            filename: Nom du fichier
            bit_depth: 8 ou 16 bits
        """
        filepath = self.output_dir / filename

        if bit_depth == 16:
            # 16-bit pour plus de précision
            data = (heightmap * 65535).astype(np.uint16)
            img = Image.fromarray(data, mode='I;16')
        else:
            # 8-bit standard
            data = (heightmap * 255).astype(np.uint8)
            img = Image.fromarray(data, mode='L')

        img.save(filepath)
        logger.info(f"Heightmap exported ({bit_depth}-bit): {filepath}")
        return str(filepath)

    def export_normal_map(
        self,
        normal_map: np.ndarray,
        filename: str = "normal_map.png"
    ):
        """
        Exporte une normal map RGB

        Args:
            normal_map: Normal map RGB (0-255 uint8)
            filename: Nom du fichier
        """
        filepath = self.output_dir / filename

        img = Image.fromarray(normal_map, mode='RGB')
        img.save(filepath)
        logger.info(f"Normal map exported: {filepath}")
        return str(filepath)

    def export_grayscale_map(
        self,
        data: np.ndarray,
        filename: str,
        bit_depth: int = 8
    ):
        """
        Exporte une map grayscale (depth, AO, roughness, etc.)

        Args:
            data: Map normalisée (0-1)
            filename: Nom du fichier
            bit_depth: 8 ou 16 bits
        """
        filepath = self.output_dir / filename

        if bit_depth == 16:
            data_uint = (data * 65535).astype(np.uint16)
            img = Image.fromarray(data_uint, mode='I;16')
        else:
            data_uint = (data * 255).astype(np.uint8)
            img = Image.fromarray(data_uint, mode='L')

        img.save(filepath)
        logger.info(f"Grayscale map exported ({bit_depth}-bit): {filepath}")
        return str(filepath)

    def export_splatmap(
        self,
        splatmap: np.ndarray,
        filename: str
    ):
        """
        Exporte une splatmap RGBA (4 channels)

        Args:
            splatmap: Splatmap RGBA (0-255 uint8)
            filename: Nom du fichier
        """
        filepath = self.output_dir / filename

        img = Image.fromarray(splatmap, mode='RGBA')
        img.save(filepath)
        logger.info(f"Splatmap exported: {filepath}")
        return str(filepath)

    def export_mesh_obj(
        self,
        heightmap: np.ndarray,
        filename: str = "terrain.obj",
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        subsample: int = 1
    ):
        """
        Exporte le terrain en mesh OBJ

        Args:
            heightmap: Heightmap (0-1)
            filename: Nom du fichier
            scale_x, scale_y, scale_z: Échelles pour X, Y (hauteur), Z
            subsample: Facteur de sous-échantillonnage (1=full, 2=half, etc.)
        """
        filepath = self.output_dir / filename

        # Sous-échantillonner si nécessaire
        if subsample > 1:
            heightmap = heightmap[::subsample, ::subsample]

        height, width = heightmap.shape

        vertices = []
        normals = []
        faces = []

        # Générer vertices
        for y in range(height):
            for x in range(width):
                vert_x = x * scale_x
                vert_y = heightmap[y, x] * scale_y
                vert_z = y * scale_z
                vertices.append((vert_x, vert_y, vert_z))

        # Générer normals (approximation)
        grad_y, grad_x = np.gradient(heightmap)
        for y in range(height):
            for x in range(width):
                nx = -grad_x[y, x] * scale_y / scale_x
                ny = 1.0
                nz = -grad_y[y, x] * scale_y / scale_z

                # Normaliser
                length = np.sqrt(nx*nx + ny*ny + nz*nz)
                normals.append((nx/length, ny/length, nz/length))

        # Générer faces (triangles)
        for y in range(height - 1):
            for x in range(width - 1):
                # Indices des vertices
                v1 = y * width + x
                v2 = y * width + (x + 1)
                v3 = (y + 1) * width + x
                v4 = (y + 1) * width + (x + 1)

                # Deux triangles par quad
                faces.append((v1, v2, v3))
                faces.append((v2, v4, v3))

        # Écrire fichier OBJ
        with open(filepath, 'w') as f:
            f.write("# Mountain Studio Pro v2.0 - Terrain Export\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n")
            f.write(f"# Resolution: {width}x{height}\n\n")

            # Vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Normals
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            f.write("\n")

            # Faces (with normals)
            for face in faces:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")

        logger.info(f"Mesh OBJ exported ({len(vertices)} vertices, {len(faces)} faces): {filepath}")
        return str(filepath)

    def export_vegetation_json(
        self,
        tree_instances: List,
        filename: str = "vegetation.json"
    ):
        """
        Exporte les instances de végétation en JSON

        Args:
            tree_instances: Liste de TreeInstance
            filename: Nom du fichier
        """
        filepath = self.output_dir / filename

        data = {
            'tree_count': len(tree_instances),
            'format_version': '2.0',
            'application': 'Mountain Studio Pro v2.0',
            'instances': [
                {
                    'position': [float(t.x), float(t.elevation), float(t.y)],
                    'species': t.species,
                    'scale': float(t.scale),
                    'rotation': float(t.rotation),
                    'age': float(t.age),
                    'health': float(t.health)
                }
                for t in tree_instances
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Vegetation JSON exported ({len(tree_instances)} instances): {filepath}")
        return str(filepath)

    def export_metadata(
        self,
        metadata: Dict,
        filename: str = "metadata.json"
    ):
        """
        Exporte les métadonnées de génération

        Args:
            metadata: Dictionnaire de métadonnées
            filename: Nom du fichier
        """
        filepath = self.output_dir / filename

        data = {
            'application': 'Mountain Studio Pro v2.0',
            'format_version': '2.0',
            **metadata
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metadata exported: {filepath}")
        return str(filepath)

    def export_vfx_prompt(
        self,
        prompt_data: Dict,
        filename: str = "vfx_prompt.txt"
    ):
        """
        Exporte le prompt VFX en format texte lisible

        Args:
            prompt_data: Données du prompt (positive, negative, metadata)
            filename: Nom du fichier
        """
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MOUNTAIN STUDIO PRO v2.0 - VFX PROMPT\n")
            f.write("="*80 + "\n\n")

            f.write("POSITIVE PROMPT:\n")
            f.write("-" * 80 + "\n")
            f.write(prompt_data.get('positive', '') + "\n\n")

            f.write("NEGATIVE PROMPT:\n")
            f.write("-" * 80 + "\n")
            f.write(prompt_data.get('negative', '') + "\n\n")

            if 'metadata' in prompt_data:
                f.write("METADATA:\n")
                f.write("-" * 80 + "\n")
                for key, value in prompt_data['metadata'].items():
                    f.write(f"{key}: {value}\n")

        logger.info(f"VFX prompt exported: {filepath}")
        return str(filepath)

    def export_complete_package(
        self,
        heightmap: np.ndarray,
        normal_map: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        ao_map: Optional[np.ndarray] = None,
        splatmaps: Optional[List[np.ndarray]] = None,
        tree_instances: Optional[List] = None,
        vfx_prompt: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        export_mesh: bool = True,
        mesh_subsample: int = 2
    ) -> Dict[str, str]:
        """
        Exporte un package complet avec tous les assets

        Args:
            heightmap: Heightmap principale
            normal_map: Normal map (optionnel)
            depth_map: Depth map (optionnel)
            ao_map: Ambient Occlusion (optionnel)
            splatmaps: Liste de splatmaps RGBA (optionnel)
            tree_instances: Instances de végétation (optionnel)
            vfx_prompt: Données du prompt VFX (optionnel)
            metadata: Métadonnées de génération (optionnel)
            export_mesh: Exporter le mesh 3D (défaut True)
            mesh_subsample: Facteur de sous-échantillonnage du mesh

        Returns:
            Dict avec paths de tous les fichiers exportés
        """
        exported_files = {}

        # Heightmap (16-bit pour qualité max)
        exported_files['heightmap'] = self.export_heightmap(heightmap, "heightmap.png", 16)

        # Normal map
        if normal_map is not None:
            exported_files['normal_map'] = self.export_normal_map(normal_map, "normal_map.png")

        # Depth map
        if depth_map is not None:
            exported_files['depth_map'] = self.export_grayscale_map(depth_map, "depth_map.png", 16)

        # AO map
        if ao_map is not None:
            exported_files['ao_map'] = self.export_grayscale_map(ao_map, "ao_map.png", 8)

        # Splatmaps
        if splatmaps is not None:
            for i, splatmap in enumerate(splatmaps):
                filename = f"splatmap_{i:02d}.png"
                exported_files[f'splatmap_{i}'] = self.export_splatmap(splatmap, filename)

        # Végétation
        if tree_instances is not None and len(tree_instances) > 0:
            exported_files['vegetation'] = self.export_vegetation_json(tree_instances, "vegetation.json")

        # VFX Prompt
        if vfx_prompt is not None:
            exported_files['vfx_prompt'] = self.export_vfx_prompt(vfx_prompt, "vfx_prompt.txt")

        # Metadata
        if metadata is not None:
            exported_files['metadata'] = self.export_metadata(metadata, "metadata.json")

        # Mesh 3D
        if export_mesh:
            exported_files['mesh_obj'] = self.export_mesh_obj(
                heightmap,
                "terrain.obj",
                scale_x=1.0,
                scale_y=50.0,  # Amplifier hauteur pour visualisation
                scale_z=1.0,
                subsample=mesh_subsample
            )

        logger.info(f"Complete package exported: {len(exported_files)} files")
        return exported_files
