"""
strategy_config_manager.py
Dynamic discovery and mapping of strategy configurations for the EV simulation framework.

Author: Eloann Le Guern--Dall'o
License: Apache-2.0
Copyright 2025 Eloann Le Guern--Dall'o
See LICENSE for details.
"""

import os
import importlib
import inspect
from typing import Dict, Callable, Optional
from utils.find_root import find_project_root
from strategies.strategy import StrategyConfig


class StrategyConfigManager:
    """
    All-in-one manager for dynamically discovering, importing, and mapping strategy configurations.
    """
    
    def __init__(self):
        self._strategy_configs: Optional[Dict[str, Callable]] = None
    
    def discover_and_map_strategies(self) -> Dict[str, Callable]:
        """
        Discover strategy files, import them, and create a mapping of strategy names to config functions.
        
        Returns:
            Dict mapping strategy class names to their default_config methods
        """
        if self._strategy_configs is not None:
            return self._strategy_configs
            
        strategy_configs = {}
        
        # Step 1: Find strategy files in the folder
        strategy_files = self._find_strategy_files()
        print(f"Found strategy files: {strategy_files}")
        
        # Step 2: Import each strategy file and extract configs
        for strategy_file in strategy_files:
            try:
                # Step 3: Dynamic import
                module = importlib.import_module(f"macmab.strategies.{strategy_file}")
                
                # Step 4: Find Config classes and map them
                configs_found = self._extract_configs_from_module(module, strategy_file)
                strategy_configs.update(configs_found)
                
            except ImportError as e:
                print(f"Warning: Could not import {strategy_file}: {e}")
            except Exception as e:
                print(f"Error processing {strategy_file}: {e}")
        
        self._strategy_configs = strategy_configs
        print(f"Successfully mapped strategies: {list(strategy_configs.keys())}")
        return strategy_configs
    
    def _find_strategy_files(self) -> list[str]:
        """
        Find all strategy Python files in the strategies folder.
        
        Returns:
            List of strategy file names (without .py extension)
        """
        strategy_names = []
        
        # Find root directory of the project
        root_dir = find_project_root()
        
        # Define the path to the strategies folder
        strategy_folder = os.path.join(root_dir, "src", "macmab", "strategies")
        
        # Check if the folder exists
        if not os.path.exists(strategy_folder):
            raise FileNotFoundError(f"Folder not found: {strategy_folder}")
        
        # List all Python files in the strategies folder
        for filename in os.listdir(strategy_folder):
            if (filename.endswith(".py") and 
                filename != "__init__.py" and 
                filename != "strategy.py"):  # Exclude base strategy file
                
                # Extract the strategy name from the filename
                strategy_name = filename[:-3]  # Remove the .py extension
                strategy_names.append(strategy_name)
        
        return strategy_names
    
    def _extract_configs_from_module(self, module, strategy_file: str) -> Dict[str, Callable]:
        """
        Extract Config classes from a strategy module.
        
        Args:
            module: The imported module
            strategy_file: Name of the strategy file (for logging)
            
        Returns:
            Dict mapping strategy names to their config functions
        """
        configs = {}
        
        # Find all classes in the module that end with 'Config'
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (name.endswith('Config') and 
                name != 'StrategyConfig' and  # Exclude base class
                hasattr(obj, 'default_config') and
                callable(getattr(obj, 'default_config'))):
                
                # Extract strategy name (remove 'Config' suffix)
                strategy_name = name[:-6]  # Remove 'Config'
                configs[strategy_name] = obj.default_config
                print(f"  Found config: {name} -> {strategy_name}")
        
        if not configs:
            print(f"  No valid config classes found in {strategy_file}")
            
        return configs
    
    def get_strategy_files(self) -> list[str]:
        """
        Get list of all strategy file names (without .py extension).
        
        Returns:
            List of strategy file names
        """
        return self._find_strategy_files()
    
    def get_available_strategies(self) -> list[str]:
        """
        Get list of all available strategy names.
        
        Returns:
            List of strategy class names
        """
        configs = self.discover_and_map_strategies()
        return list(configs.keys())
    
    def set_strategy_config(self, strategy_name: str) -> StrategyConfig:
        """
        Get the strategy config for the given strategy name.
        
        Args:
            strategy_name: Name of the strategy (class name, e.g., 'Baseline', 'CMABCLinTS')
            
        Returns:
            StrategyConfig object
            
        Raises:
            ValueError: If strategy name is not found
        """
        strategy_configs = self.discover_and_map_strategies()
        
        config_func = strategy_configs.get(strategy_name)
        if config_func is None:
            available_strategies = list(strategy_configs.keys())
            raise ValueError(
                f"Unknown strategy name: '{strategy_name}'. "
                f"Available strategies: {available_strategies}"
            )
        
        return config_func()


# Global instance - create once and reuse
_strategy_manager = StrategyConfigManager()

# Convenience functions for backward compatibility
def discover_and_map_strategies() -> Dict[str, Callable]:
    """Discover and map all strategies. Returns mapping of strategy names to config functions."""
    return _strategy_manager.discover_and_map_strategies()

def get_strategy_files() -> list[str]:
    """Get list of all strategy file names."""
    return _strategy_manager.get_strategy_files()

def get_available_strategies() -> list[str]:
    """Get list of all available strategy names."""
    return _strategy_manager.get_available_strategies()

def set_strategy_config(strategy_name: str) -> StrategyConfig:
    """
    Set dynamically the strategy config based on the strategy name.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'Baseline', 'CMABCLinTS')
        
    Returns:
        StrategyConfig object
    """
    return _strategy_manager.set_strategy_config(strategy_name)


# Usage in your class:
class YourSimulationClass:
    def set_strategy_config(self) -> StrategyConfig:
        """
        Set dynamically the strategy config based on the strategy name in the simulation config.
        Returns the appropriate StrategyConfig object.
        """
        return set_strategy_config(self.config.strategy_name)