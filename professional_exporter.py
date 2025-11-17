"""
Export professionnel pour graphistes
Support EXR 32-bit, multi-channel, formats standards de l'industrie
"""

import numpy as np
from PIL import Image
import os
from typing import Dict, Optional, List
import struct


class ProfessionalExporter:
    """Exporte les assets dans des formats professionnels"""

    def __init__(self):
        self.supported_formats = ['png', 'exr', 'tiff', 'tga']

    def export_heightmap_exr(self, heightmap: np.ndarray, filepath: str):
        """
        Exporte heightmap en EXR 32-bit float
        Format standard pour displacement en production
        """
        try:
            import OpenEXR
            import Imath

            height, width = heightmap.shape

            # Convertir en float32
            data = heightmap.astype(np.float32)

            # Créer le header EXR
            header = OpenEXR.Header(width, height)
            header['channels'] = {
                'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }

            # Créer le fichier
            exr_file = OpenEXR.OutputFile(filepath, header)
            exr_file.writePixels({'Y': data.tobytes()})
            exr_file.close()

            print(f"✓ Heightmap EXR exporté: {filepath}")
            return True

        except ImportError:
            print("⚠ OpenEXR non installé, export en TIFF 32-bit à la place")
            return self.export_heightmap_tiff32(heightmap, filepath.replace('.exr', '.tiff'))

    def export_heightmap_tiff32(self, heightmap: np.ndarray, filepath: str):
        """Exporte heightmap en TIFF 32-bit float"""
        from PIL import Image
        import tifffile

        try:
            # Sauver en TIFF 32-bit float
            tifffile.imwrite(filepath, heightmap.astype(np.float32))
            print(f"✓ Heightmap TIFF 32-bit exporté: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Erreur export TIFF: {e}")
            # Fallback PNG 16-bit
            heightmap_16bit = (heightmap * 65535).astype(np.uint16)
            img = Image.fromarray(heightmap_16bit, mode='I;16')
            img.save(filepath.replace('.tiff', '_16bit.png'))
            print(f"✓ Fallback: Heightmap PNG 16-bit exporté")
            return True

    def export_normal_map(self, normal_map: np.ndarray, filepath: str, format: str = 'png'):
        """
        Exporte normal map dans différents formats
        Formats: DirectX (R,G,B), OpenGL (R,G,-B)
        """
        if format.lower() == 'png':
            img = Image.fromarray(normal_map, mode='RGB')
            img.save(filepath)
        elif format.lower() == 'tga':
            # TGA pour compatibilité game engines
            img = Image.fromarray(normal_map, mode='RGB')
            img.save(filepath, format='TGA')
        elif format.lower() == 'exr':
            # Normal map en EXR pour precision maximale
            self._export_rgb_exr(normal_map, filepath)

        print(f"✓ Normal map exportée: {filepath}")

    def export_pbr_maps(self,
                       base_color: Optional[np.ndarray],
                       normal: Optional[np.ndarray],
                       roughness: Optional[np.ndarray],
                       metallic: Optional[np.ndarray],
                       ao: Optional[np.ndarray],
                       height: Optional[np.ndarray],
                       output_dir: str,
                       prefix: str = "material"):
        """
        Exporte un set complet de textures PBR
        Compatible avec Unreal, Unity, Blender, etc.
        """
        os.makedirs(output_dir, exist_ok=True)

        maps = {
            'BaseColor': (base_color, 'RGB'),
            'Normal': (normal, 'RGB'),
            'Roughness': (roughness, 'L'),
            'Metallic': (metallic, 'L'),
            'AO': (ao, 'L'),
            'Height': (height, 'L')
        }

        for map_name, (data, mode) in maps.items():
            if data is not None:
                filepath = os.path.join(output_dir, f"{prefix}_{map_name}.png")

                if data.dtype in [np.float32, np.float64]:
                    data = (data * 255).astype(np.uint8)

                if mode == 'L' and len(data.shape) == 2:
                    img = Image.fromarray(data, mode='L')
                elif mode == 'RGB':
                    img = Image.fromarray(data, mode='RGB')
                else:
                    continue

                img.save(filepath)
                print(f"✓ {map_name} exporté")

        # Créer un fichier metadata
        self._create_pbr_metadata(output_dir, prefix, maps.keys())

    def export_to_substance(self, maps: Dict[str, np.ndarray], output_dir: str, prefix: str):
        """
        Exporte dans un format compatible Substance Painter/Designer
        """
        # Substance attend certains noms spécifiques
        substance_mapping = {
            'heightmap': 'Height',
            'normal_map': 'Normal',
            'ao_map': 'AmbientOcclusion',
            'roughness_map': 'Roughness',
            'base_color': 'BaseColor'
        }

        os.makedirs(output_dir, exist_ok=True)

        for internal_name, substance_name in substance_mapping.items():
            if internal_name in maps and maps[internal_name] is not None:
                data = maps[internal_name]

                # Substance préfère TIFF 16-bit ou PNG
                filepath = os.path.join(output_dir, f"{prefix}_{substance_name}.tif")

                if data.dtype in [np.float32, np.float64]:
                    # Convertir en 16-bit
                    data_16bit = (data * 65535).astype(np.uint16)
                    img = Image.fromarray(data_16bit)
                else:
                    img = Image.fromarray(data)

                img.save(filepath, compression='tiff_deflate')
                print(f"✓ {substance_name} exporté pour Substance")

    def export_to_blender(self, terrain_gen, output_dir: str, prefix: str = "mountain"):
        """
        Exporte tout ce qui est nécessaire pour Blender
        - Heightmap (EXR/PNG 16-bit pour displacement)
        - Normal map
        - Textures
        - Mesh OBJ
        """
        os.makedirs(output_dir, exist_ok=True)

        # Heightmap pour displacement
        heightmap_path = os.path.join(output_dir, f"{prefix}_displacement.exr")
        self.export_heightmap_exr(terrain_gen.heightmap, heightmap_path)

        # Normal map
        if terrain_gen.normal_map is not None:
            normal_path = os.path.join(output_dir, f"{prefix}_normal.png")
            self.export_normal_map(terrain_gen.normal_map, normal_path)

        # Mesh (optionnel, pour prévisualisation)
        mesh_path = os.path.join(output_dir, f"{prefix}_mesh.obj")
        vertices, faces, normals = terrain_gen.get_3d_mesh_data()
        self._export_obj(mesh_path, vertices, faces)

        # Créer un script Python Blender pour auto-setup
        self._create_blender_script(output_dir, prefix)

        print(f"✓ Export Blender complet dans: {output_dir}")

    def export_to_unreal(self, maps: Dict[str, np.ndarray], output_dir: str, prefix: str):
        """
        Exporte pour Unreal Engine
        - Heightmap (PNG 16-bit ou RAW)
        - Normal map (format UE)
        - Packed textures (R=Metallic, G=Roughness, B=AO)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Heightmap 16-bit
        if 'heightmap' in maps:
            heightmap_16 = (maps['heightmap'] * 65535).astype(np.uint16)
            heightmap_path = os.path.join(output_dir, f"{prefix}_Heightmap.png")
            img = Image.fromarray(heightmap_16, mode='I;16')
            img.save(heightmap_path)
            print(f"✓ Heightmap UE exporté")

        # Normal map (DirectX format pour UE)
        if 'normal_map' in maps:
            normal_path = os.path.join(output_dir, f"{prefix}_Normal.png")
            self.export_normal_map(maps['normal_map'], normal_path)

        # Packed texture (ORM - Occlusion, Roughness, Metallic)
        if all(k in maps for k in ['ao_map', 'roughness_map']):
            packed = np.zeros((*maps['ao_map'].shape, 3), dtype=np.uint8)
            packed[:, :, 0] = maps['ao_map']  # R = AO
            packed[:, :, 1] = maps['roughness_map']  # G = Roughness
            # B = Metallic (default 0 pour terrain naturel)

            packed_path = os.path.join(output_dir, f"{prefix}_ORM.png")
            img = Image.fromarray(packed, mode='RGB')
            img.save(packed_path)
            print(f"✓ Packed ORM texture exportée")

        print(f"✓ Export Unreal Engine complet")

    def _export_rgb_exr(self, rgb_data: np.ndarray, filepath: str):
        """Exporte RGB en EXR"""
        try:
            import OpenEXR
            import Imath

            height, width, channels = rgb_data.shape

            # Convertir en float32
            r = (rgb_data[:, :, 0] / 255.0).astype(np.float32)
            g = (rgb_data[:, :, 1] / 255.0).astype(np.float32)
            b = (rgb_data[:, :, 2] / 255.0).astype(np.float32)

            header = OpenEXR.Header(width, height)
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }

            exr_file = OpenEXR.OutputFile(filepath, header)
            exr_file.writePixels({
                'R': r.tobytes(),
                'G': g.tobytes(),
                'B': b.tobytes()
            })
            exr_file.close()

        except ImportError:
            # Fallback PNG
            img = Image.fromarray(rgb_data, mode='RGB')
            img.save(filepath.replace('.exr', '.png'))

    def _export_obj(self, filepath: str, vertices: np.ndarray, faces: np.ndarray):
        """Exporte mesh en OBJ"""
        with open(filepath, 'w') as f:
            f.write("# Mountain Studio Pro - 3D Terrain Mesh\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")

            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def _create_pbr_metadata(self, output_dir: str, prefix: str, map_names: List[str]):
        """Crée un fichier metadata pour les textures PBR"""
        metadata_path = os.path.join(output_dir, f"{prefix}_metadata.txt")

        with open(metadata_path, 'w') as f:
            f.write("# Mountain Studio Pro - PBR Material Metadata\n")
            f.write(f"# Material: {prefix}\n\n")
            f.write("Maps included:\n")
            for name in map_names:
                f.write(f"  - {name}\n")
            f.write("\nWorkflow: PBR Metallic/Roughness\n")
            f.write("Color Space: sRGB (BaseColor), Linear (other maps)\n")
            f.write("Normal Map Format: OpenGL (Y+)\n")

    def _create_blender_script(self, output_dir: str, prefix: str):
        """Crée un script Python pour importer automatiquement dans Blender"""
        script_path = os.path.join(output_dir, f"{prefix}_blender_import.py")

        script_content = f"""
# Mountain Studio Pro - Auto Import Script pour Blender
# Exécutez ce script dans Blender pour configurer automatiquement le terrain

import bpy
import os

# Chemin du dossier
basepath = r"{output_dir}"

# Créer un nouveau mesh plane
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "{prefix}_Terrain"

# Subdiviser
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=200)
bpy.ops.object.mode_set(mode='OBJECT')

# Ajouter modificateur displacement
disp_mod = plane.modifiers.new(name="Displacement", type='DISPLACE')

# Créer texture displacement
disp_tex = bpy.data.textures.new(name="{prefix}_Height", type='IMAGE')
disp_tex.image = bpy.data.images.load(os.path.join(basepath, "{prefix}_displacement.exr"))
disp_mod.texture = disp_tex
disp_mod.strength = 2.0

# Setup material avec normal map
mat = bpy.data.materials.new(name="{prefix}_Material")
mat.use_nodes = True
plane.data.materials.append(mat)

nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Charger normal map
normal_tex = nodes.new('ShaderNodeTexImage')
normal_tex.image = bpy.data.images.load(os.path.join(basepath, "{prefix}_normal.png"))
normal_tex.image.colorspace_settings.name = 'Non-Color'

# Normal map node
normal_map = nodes.new('ShaderNodeNormalMap')
links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])

# Connecter au BSDF
bsdf = nodes['Principled BSDF']
links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

print("✓ Terrain '{prefix}' importé avec succès!")
"""

        with open(script_path, 'w') as f:
            f.write(script_content)

        print(f"✓ Script Blender créé: {script_path}")

    def create_contact_sheet(self, maps: Dict[str, np.ndarray], output_path: str):
        """
        Crée une planche contact avec toutes les maps
        Utile pour validation rapide
        """
        from PIL import Image, ImageDraw, ImageFont

        # Calculer layout
        n_maps = len([m for m in maps.values() if m is not None])
        cols = 3
        rows = (n_maps + cols - 1) // cols

        preview_size = 512
        margin = 20
        label_height = 30

        sheet_width = cols * (preview_size + margin) + margin
        sheet_height = rows * (preview_size + label_height + margin) + margin

        # Créer image
        contact_sheet = Image.new('RGB', (sheet_width, sheet_height), color=(40, 40, 40))
        draw = ImageDraw.Draw(contact_sheet)

        # Placer les maps
        idx = 0
        for map_name, map_data in maps.items():
            if map_data is None:
                continue

            row = idx // cols
            col = idx % cols

            x = col * (preview_size + margin) + margin
            y = row * (preview_size + label_height + margin) + margin

            # Convertir map en image
            if len(map_data.shape) == 2:
                if map_data.dtype in [np.float32, np.float64]:
                    map_data = (map_data * 255).astype(np.uint8)
                map_img = Image.fromarray(map_data, mode='L').convert('RGB')
            else:
                map_img = Image.fromarray(map_data, mode='RGB')

            # Resize
            map_img = map_img.resize((preview_size, preview_size), Image.Resampling.LANCZOS)

            # Coller
            contact_sheet.paste(map_img, (x, y + label_height))

            # Label
            draw.text((x, y), map_name, fill=(255, 255, 255))

            idx += 1

        contact_sheet.save(output_path)
        print(f"✓ Planche contact créée: {output_path}")
