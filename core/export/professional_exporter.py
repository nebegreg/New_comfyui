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
        subsample: int = 1,
        with_mtl: bool = False,
        mtl_name: str = "terrain"
    ):
        """
        Exporte le terrain en mesh OBJ avec UVs et optionnel MTL

        Args:
            heightmap: Heightmap (0-1)
            filename: Nom du fichier
            scale_x, scale_y, scale_z: Échelles pour X, Y (hauteur), Z
            subsample: Facteur de sous-échantillonnage (1=full, 2=half, etc.)
            with_mtl: Générer fichier MTL associé
            mtl_name: Nom du matériau
        """
        filepath = self.output_dir / filename

        # Sous-échantillonner si nécessaire
        if subsample > 1:
            heightmap = heightmap[::subsample, ::subsample]

        height, width = heightmap.shape

        vertices = []
        normals = []
        uvs = []
        faces = []

        # Générer vertices et UVs
        for y in range(height):
            for x in range(width):
                vert_x = x * scale_x
                vert_y = heightmap[y, x] * scale_y
                vert_z = y * scale_z
                vertices.append((vert_x, vert_y, vert_z))

                # UVs (0-1)
                u = x / (width - 1)
                v = y / (height - 1)
                uvs.append((u, v))

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

        # Générer faces (triangles) avec UVs
        for y in range(height - 1):
            for x in range(width - 1):
                # Indices des vertices
                v1 = y * width + x
                v2 = y * width + (x + 1)
                v3 = (y + 1) * width + x
                v4 = (y + 1) * width + (x + 1)

                # Deux triangles par quad (v/vt/vn format)
                faces.append((v1, v2, v3))
                faces.append((v2, v4, v3))

        # Écrire fichier OBJ
        with open(filepath, 'w') as f:
            f.write("# Mountain Studio Pro v2.0 - Terrain Export\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n")
            f.write(f"# Resolution: {width}x{height}\n")

            if with_mtl:
                mtl_filename = filename.replace('.obj', '.mtl')
                f.write(f"mtllib {mtl_filename}\n")
                f.write(f"usemtl {mtl_name}\n")

            f.write("\n")

            # Vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # UVs
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            f.write("\n")

            # Normals
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            f.write("\n")

            # Faces (v/vt/vn format)
            for face in faces:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")

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

    def export_mtl_file(
        self,
        mtl_filename: str,
        material_name: str,
        diffuse_map: Optional[str] = None,
        normal_map: Optional[str] = None,
        roughness_map: Optional[str] = None,
        ao_map: Optional[str] = None,
        displacement_map: Optional[str] = None
    ) -> str:
        """
        Crée un fichier MTL (Material) pour OBJ

        Args:
            mtl_filename: Nom du fichier MTL
            material_name: Nom du matériau
            diffuse_map: Chemin texture diffuse/albedo (relatif)
            normal_map: Chemin normal map (relatif)
            roughness_map: Chemin roughness map (relatif)
            ao_map: Chemin AO map (relatif)
            displacement_map: Chemin displacement/heightmap (relatif)

        Returns:
            Path du fichier MTL créé
        """
        filepath = self.output_dir / mtl_filename

        with open(filepath, 'w') as f:
            f.write("# Mountain Studio Pro v2.0 - Material File\n")
            f.write(f"# Created for Autodesk Flame compatibility\n\n")

            f.write(f"newmtl {material_name}\n")

            # Propriétés de base
            f.write("Ka 1.0 1.0 1.0\n")  # Ambient color
            f.write("Kd 0.8 0.8 0.8\n")  # Diffuse color
            f.write("Ks 0.5 0.5 0.5\n")  # Specular color
            f.write("Ns 96.0\n")           # Specular exponent
            f.write("d 1.0\n")              # Dissolve (opacity)
            f.write("illum 2\n")            # Illumination model

            # Texture maps
            if diffuse_map:
                f.write(f"map_Kd {diffuse_map}\n")  # Diffuse/Albedo texture

            if normal_map:
                f.write(f"map_Bump {normal_map}\n")  # Normal map
                f.write(f"bump {normal_map}\n")

            if roughness_map:
                f.write(f"map_Ns {roughness_map}\n")  # Roughness/Specular map

            if ao_map:
                f.write(f"map_Ka {ao_map}\n")  # AO map

            if displacement_map:
                f.write(f"disp {displacement_map}\n")  # Displacement map

        logger.info(f"MTL file exported: {filepath}")
        return str(filepath)

    def export_for_autodesk_flame(
        self,
        heightmap: np.ndarray,
        normal_map: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        ao_map: Optional[np.ndarray] = None,
        diffuse_map: Optional[np.ndarray] = None,
        roughness_map: Optional[np.ndarray] = None,
        splatmaps: Optional[List[np.ndarray]] = None,
        tree_instances: Optional[List] = None,
        mesh_subsample: int = 2,
        scale_y: float = 50.0
    ) -> Dict[str, str]:
        """
        Export complet optimisé pour Autodesk Flame

        Structure exportée:
        terrain_export/
            terrain.obj           # Mesh avec UVs
            terrain.mtl           # Matériau
            textures/
                diffuse.png       # Albedo/Diffuse
                normal.png        # Normal map
                roughness.png     # Roughness
                ao.png            # Ambient Occlusion
                height.png        # Displacement
                depth.png         # Z-depth
            vegetation.json       # Instances d'arbres

        Args:
            heightmap: Heightmap (0-1)
            normal_map: Normal map RGB
            depth_map: Depth map
            ao_map: Ambient Occlusion
            diffuse_map: Diffuse/Albedo map
            roughness_map: Roughness map
            splatmaps: PBR splatmaps
            tree_instances: Végétation
            mesh_subsample: Facteur sous-échantillonnage mesh
            scale_y: Amplification hauteur

        Returns:
            Dict avec tous les fichiers exportés
        """
        exported_files = {}

        # Créer sous-dossier textures
        textures_dir = self.output_dir / "textures"
        textures_dir.mkdir(exist_ok=True)

        logger.info("=== Export pour Autodesk Flame ===")

        # 1. Exporter toutes les textures dans /textures/
        texture_paths = {}

        # Heightmap (displacement)
        height_path = textures_dir / "height.png"
        height_img = Image.fromarray((heightmap * 65535).astype(np.uint16), mode='I;16')
        height_img.save(height_path)
        texture_paths['displacement'] = "textures/height.png"
        exported_files['displacement'] = str(height_path)
        logger.info(f"  ✓ Displacement map: {height_path}")

        # Normal map
        if normal_map is not None:
            normal_path = textures_dir / "normal.png"
            normal_img = Image.fromarray(normal_map, mode='RGB')
            normal_img.save(normal_path)
            texture_paths['normal'] = "textures/normal.png"
            exported_files['normal'] = str(normal_path)
            logger.info(f"  ✓ Normal map: {normal_path}")

        # Depth map
        if depth_map is not None:
            depth_path = textures_dir / "depth.png"
            # Convert to float first to avoid overflow, then to uint16
            depth_normalized = depth_map.astype(np.float32)
            if depth_normalized.max() > 1.0:
                depth_normalized = depth_normalized / 255.0  # Normalize if in 0-255 range
            depth_img = Image.fromarray((depth_normalized * 65535).astype(np.uint16), mode='I;16')
            depth_img.save(depth_path)
            texture_paths['depth'] = "textures/depth.png"
            exported_files['depth'] = str(depth_path)
            logger.info(f"  ✓ Depth map: {depth_path}")

        # AO map
        if ao_map is not None:
            ao_path = textures_dir / "ao.png"
            # Normalize if needed
            ao_normalized = ao_map.astype(np.float32)
            if ao_normalized.max() > 1.0:
                ao_normalized = ao_normalized / 255.0
            ao_img = Image.fromarray((ao_normalized * 255).astype(np.uint8), mode='L')
            ao_img.save(ao_path)
            texture_paths['ao'] = "textures/ao.png"
            exported_files['ao'] = str(ao_path)
            logger.info(f"  ✓ AO map: {ao_path}")

        # Diffuse map (si fournie ou générer depuis heightmap)
        if diffuse_map is not None:
            diffuse_path = textures_dir / "diffuse.png"
            if len(diffuse_map.shape) == 2:
                # Grayscale diffuse
                diffuse_normalized = diffuse_map.astype(np.float32)
                if diffuse_normalized.max() > 1.0:
                    diffuse_normalized = diffuse_normalized / 255.0
                diffuse_img = Image.fromarray((diffuse_normalized * 255).astype(np.uint8), mode='L')
            else:
                # RGB diffuse - assume already in 0-255 range
                diffuse_img = Image.fromarray(diffuse_map.astype(np.uint8), mode='RGB')
            diffuse_img.save(diffuse_path)
            texture_paths['diffuse'] = "textures/diffuse.png"
            exported_files['diffuse'] = str(diffuse_path)
            logger.info(f"  ✓ Diffuse map: {diffuse_path}")
        else:
            # Générer diffuse basique depuis heightmap (grayscale)
            diffuse_path = textures_dir / "diffuse.png"
            heightmap_normalized = heightmap.astype(np.float32)
            if heightmap_normalized.max() > 1.0:
                heightmap_normalized = heightmap_normalized / 255.0
            diffuse_img = Image.fromarray((heightmap_normalized * 255).astype(np.uint8), mode='L')
            diffuse_img.save(diffuse_path)
            texture_paths['diffuse'] = "textures/diffuse.png"
            exported_files['diffuse'] = str(diffuse_path)
            logger.info(f"  ✓ Diffuse map (generated): {diffuse_path}")

        # Roughness map
        if roughness_map is not None:
            roughness_path = textures_dir / "roughness.png"
            # Normalize if needed
            roughness_normalized = roughness_map.astype(np.float32)
            if roughness_normalized.max() > 1.0:
                roughness_normalized = roughness_normalized / 255.0
            roughness_img = Image.fromarray((roughness_normalized * 255).astype(np.uint8), mode='L')
            roughness_img.save(roughness_path)
            texture_paths['roughness'] = "textures/roughness.png"
            exported_files['roughness'] = str(roughness_path)
            logger.info(f"  ✓ Roughness map: {roughness_path}")

        # Splatmaps
        if splatmaps is not None:
            for i, splatmap in enumerate(splatmaps):
                splatmap_path = textures_dir / f"splatmap_{i:02d}.png"
                splatmap_img = Image.fromarray(splatmap, mode='RGBA')
                splatmap_img.save(splatmap_path)
                exported_files[f'splatmap_{i}'] = str(splatmap_path)
                logger.info(f"  ✓ Splatmap {i}: {splatmap_path}")

        # 2. Créer fichier MTL
        mtl_path = self.export_mtl_file(
            "terrain.mtl",
            "terrain_material",
            diffuse_map=texture_paths.get('diffuse'),
            normal_map=texture_paths.get('normal'),
            roughness_map=texture_paths.get('roughness'),
            ao_map=texture_paths.get('ao'),
            displacement_map=texture_paths.get('displacement')
        )
        exported_files['mtl'] = mtl_path
        logger.info(f"  ✓ MTL file: {mtl_path}")

        # 3. Exporter mesh OBJ avec MTL
        obj_path = self.export_mesh_obj(
            heightmap,
            "terrain.obj",
            scale_x=1.0,
            scale_y=scale_y,
            scale_z=1.0,
            subsample=mesh_subsample,
            with_mtl=True,
            mtl_name="terrain_material"
        )
        exported_files['obj'] = obj_path
        logger.info(f"  ✓ OBJ file: {obj_path}")

        # 4. Exporter végétation si disponible
        if tree_instances is not None and len(tree_instances) > 0:
            veg_path = self.export_vegetation_json(tree_instances, "vegetation.json")
            exported_files['vegetation'] = veg_path
            logger.info(f"  ✓ Vegetation: {veg_path} ({len(tree_instances)} instances)")

        # 5. Créer fichier README pour Flame
        readme_path = self.output_dir / "README_FLAME.txt"
        with open(readme_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MOUNTAIN STUDIO PRO v2.0 - AUTODESK FLAME EXPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("FICHIERS EXPORTÉS:\n")
            f.write("-" * 80 + "\n")
            f.write("terrain.obj          - Mesh 3D principal avec UVs\n")
            f.write("terrain.mtl          - Fichier matériau (référence les textures)\n")
            f.write("\nTEXTURES (dossier textures/):\n")
            f.write("-" * 80 + "\n")
            f.write("diffuse.png          - Albedo/Diffuse map\n")
            if 'normal' in exported_files:
                f.write("normal.png           - Normal map (bump mapping)\n")
            if 'roughness' in exported_files:
                f.write("roughness.png        - Roughness map (PBR)\n")
            if 'ao' in exported_files:
                f.write("ao.png               - Ambient Occlusion\n")
            f.write("height.png           - Displacement map (16-bit)\n")
            if 'depth' in exported_files:
                f.write("depth.png            - Z-depth map (16-bit)\n")

            if splatmaps:
                f.write(f"\n{len(splatmaps)} splatmaps pour matériaux multiples\n")

            f.write("\nIMPORT DANS FLAME:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Importer terrain.obj dans votre scène 3D\n")
            f.write("2. Le fichier .mtl sera automatiquement détecté\n")
            f.write("3. Les textures sont référencées relativement (./textures/)\n")
            f.write("4. Ajuster l'échelle si nécessaire (hauteur = {:.1f}x)\n".format(scale_y))

            if tree_instances:
                f.write("\nVÉGÉTATION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"vegetation.json contient {len(tree_instances)} instances d'arbres\n")
                f.write("Format: position [x, height, z], species, scale, rotation, age, health\n")
                f.write("Peut être importé via script Python/Action dans Flame\n")

            f.write("\n" + "=" * 80 + "\n")

        exported_files['readme'] = str(readme_path)
        logger.info(f"  ✓ README: {readme_path}")

        logger.info(f"=== Export Flame terminé: {len(exported_files)} fichiers ===")
        return exported_files
