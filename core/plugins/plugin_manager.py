"""
Plugin System for Mountain Studio
==================================

Extensibility via plugins.

Author: Mountain Studio Pro Team
"""

import logging
from pathlib import Path
from typing import Dict, List, Callable, Any
import importlib.util

logger = logging.getLogger(__name__)


class PluginHook:
    """Plugin hook point"""
    def __init__(self, name: str):
        self.name = name
        self.callbacks: List[Callable] = []
        
    def register(self, callback: Callable):
        """Register callback"""
        self.callbacks.append(callback)
        
    def execute(self, *args, **kwargs) -> List[Any]:
        """Execute all callbacks"""
        results = []
        for callback in self.callbacks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {self.name} callback failed: {e}")
        return results


class PluginManager:
    """Plugin manager"""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        
        self.plugins: Dict[str, Any] = {}
        self.hooks: Dict[str, PluginHook] = {}
        
        # Register standard hooks
        self._register_standard_hooks()
        
        logger.info(f"PluginManager initialized: {plugin_dir}")
        
    def _register_standard_hooks(self):
        """Register standard hook points"""
        hooks = [
            'on_terrain_generated',
            'on_vegetation_generated',
            'on_pbr_generated',
            'on_hdri_generated',
            'custom_export_format',
            'custom_ui_tab'
        ]
        
        for hook_name in hooks:
            self.hooks[hook_name] = PluginHook(hook_name)
            
    def load_plugins(self):
        """Load all plugins from plugin directory"""
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.stem.startswith("_"):
                continue
                
            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
                
    def _load_plugin(self, plugin_path: Path):
        """Load single plugin"""
        spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Plugin must have setup() function
        if hasattr(module, 'setup'):
            module.setup(self)
            self.plugins[plugin_path.stem] = module
            logger.info(f"Loaded plugin: {plugin_path.stem}")
        else:
            logger.warning(f"Plugin {plugin_path.stem} has no setup() function")
            
    def get_hook(self, name: str) -> PluginHook:
        """Get hook by name"""
        if name not in self.hooks:
            self.hooks[name] = PluginHook(name)
        return self.hooks[name]
        
    def execute_hook(self, name: str, *args, **kwargs) -> List[Any]:
        """Execute hook"""
        hook = self.get_hook(name)
        return hook.execute(*args, **kwargs)


# Global plugin manager
_plugin_manager = None

def get_plugin_manager() -> PluginManager:
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
