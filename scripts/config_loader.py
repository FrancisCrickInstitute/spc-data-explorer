"""
Singleton configuration loader for the Phenotype Clustering Interactive Visualization App.

This module provides interactive config selection with singleton pattern to avoid multiple prompts.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, List, Optional


class ConfigLoader:
    """Handles loading different configuration files with interactive selection."""
    
    _config = None  # Singleton instance
    _config_loaded = False
    
    @classmethod
    def list_available_configs(cls) -> List[str]:
        """List all available configuration files."""
        config_dir = Path(__file__).parent / "config"
        if not config_dir.exists():
            print(" Config directory not found!")
            return []
        
        config_files = []
        for file in config_dir.glob("config_*.py"):
            config_name = file.stem  # Remove .py extension
            config_files.append(config_name)
        
        return sorted(config_files)
    
    @classmethod
    def interactive_config_selection(cls) -> str:
        """Interactive configuration selection."""
        configs = cls.list_available_configs()
        
        if not configs:
            print(" No configuration files found in config/ directory")
            print("Create configuration files like: config/config_dataset1.py")
            sys.exit(1)
        
        print(f"\n🔧 Available Configurations:")
        print("=" * 50)
        for i, config in enumerate(configs, 1):
            # Clean up display name
            display_name = config.replace('config_', '').replace('_', ' ').title()
            print(f"{i:2d}. {display_name} ({config})")
        
        print("=" * 50)
        
        while True:
            try:
                choice = input(f"Select configuration (1-{len(configs)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print(" Goodbye!")
                    sys.exit(0)
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(configs):
                    selected_config = configs[choice_idx]
                    print(f" Selected: {selected_config}")
                    return selected_config
                else:
                    print(" Invalid selection! Please try again.")
                    
            except (ValueError, KeyboardInterrupt):
                print("\n Goodbye!")
                sys.exit(0)
    
    @classmethod
    def load_config(cls, config_name: str = None, interactive: bool = True) -> Any:
        """
        Load configuration from specified config file (singleton pattern).
        
        Args:
            config_name: Name of config file (without .py extension)
            interactive: If True, show interactive selection when config_name is None
            
        Returns:
            Config class from the loaded module
        """
        # Return cached config if already loaded
        if cls._config_loaded and cls._config is not None:
            return cls._config
        
        # Determine config name from various sources
        if not config_name:
            # Try command line argument
            if len(sys.argv) > 2 and sys.argv[1] == '--config':
                config_name = sys.argv[2]
                # Remove from sys.argv so it doesn't interfere with other argument parsing
                sys.argv = sys.argv[:1] + sys.argv[3:]
            
            # Try environment variable
            elif 'SPC_CONFIG' in os.environ:
                config_name = os.environ['SPC_CONFIG']
            
            # Interactive selection
            elif interactive:
                config_name = cls.interactive_config_selection()
            
            # Default fallback
            else:
                config_name = 'config_default'
        
        # Construct path to config file
        config_dir = Path(__file__).parent / "config"
        config_file = config_dir / f"{config_name}.py"
        
        if not config_file.exists():
            print(f" Configuration file not found: {config_file}")
            available_configs = cls.list_available_configs()
            if available_configs:
                print(f"Available configs in {config_dir}:")
                for cfg in available_configs:
                    print(f"  • {cfg}")
            
            if interactive and not cls._config_loaded:
                print("\nWould you like to select from available configs?")
                choice = input("Press Enter to select or 'q' to quit: ").strip()
                if choice.lower() != 'q':
                    return cls.load_config(interactive=True)
            
            sys.exit(1)
        
        # Load the config module dynamically
        try:
            spec = importlib.util.spec_from_file_location("config_module", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Validate the config has required attributes
            if not hasattr(config_module, 'Config'):
                print(f" Configuration file {config_name} is missing 'Config' class!")
                sys.exit(1)
            
            print(f" Loaded configuration: {config_name}")
            
            # Cache the config (singleton pattern)
            cls._config = config_module.Config
            cls._config_loaded = True
            
            return cls._config
            
        except Exception as e:
            print(f" Error loading configuration {config_name}: {e}")
            sys.exit(1)


def get_config() -> Any:
    """
    Convenience function to get config with interactive selection.
    Uses singleton pattern to avoid multiple prompts.
    """
    return ConfigLoader.load_config()