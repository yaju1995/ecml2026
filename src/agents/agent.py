import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from tabulate import tabulate


@dataclass
class AgentConfig:
    """Configuration class for Agent parameters."""

    type: str  # Agent type
    id: int  # Agent ID

    @classmethod
    def get_config_dict(cls, config):
        """Return the configuration as a dictionary"""
        return config.__dict__


@dataclass
class AgentSize:
    """Data class with size methods."""

    def size(self) -> int:
        """Return the number of fields in the state."""
        return len(fields(self))


class Agent(ABC):
    """Abstract base class for Agent implementations."""

    def __init__(self, config: AgentConfig):
        """Initialize base Agent parameters."""
        # Core parameters
        self.id = config.id
        self.type = config.type

        # Configuration
        self._set_config(config)

    def _set_config(self, config: AgentConfig):
        """Set configuration parameters."""
        for key, value in vars(config).items():
            setattr(self, key, value)

    @abstractmethod
    def reset(self):
        """Reset agent state."""
        pass

    @abstractmethod
    def act(self, state: np.ndarray):
        """Select an action given the current state."""
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Update agent state given the observed transition."""
        pass


class AgentInfoMixin:
    """Mixin class providing standardized agent information access."""

    def get_info(self) -> Dict[str, Any]:
        """Get all available agent information as a dictionary."""
        return {
            "core": self.get_core_info(),
            "config": self.get_config_info(),
            "state": self.get_state_info(),
            "observation": self.get_observation_info(),
        }

    def get_core_info(self) -> Dict[str, Any]:
        """Get core agent information."""
        return {"id": self.id, "type": self.type}

    def get_config_info(self) -> Dict[str, Any]:
        """Get agent configuration information."""
        if hasattr(self, "config"):
            return asdict(self.config)
        return {
            k: v for k, v in vars(self).items() if not k.startswith("_") and k not in ["id", "type"]
        }

    def get_state_info(self) -> Dict[str, Any]:
        """Get current agent state information."""
        state_attrs = self._get_state_attributes()
        return {k: getattr(self, k) for k in state_attrs if hasattr(self, k)}

    def get_observation_info(self) -> Dict[str, Any]:
        """Get current observation information."""
        obs_attrs = self._get_observation_attributes()
        return {k: getattr(self, k) for k in obs_attrs if hasattr(self, k)}

    def _get_state_attributes(self) -> List[str]:
        """Get list of state-related attributes."""
        # Override this method in concrete agent classes to specify state attributes
        return []

    def query_info(self, keys: List[str]) -> Dict[str, Any]:
        """
        Query specific information using dot notation.
        Example: ['core.id', 'config.dt', 'state.position']
        """
        result = {}
        info = self.get_info()

        for key in keys:
            try:
                sections = key.split(".")
                value = info
                for section in sections:
                    value = value[section]
                result[key] = value
            except (KeyError, TypeError):
                result[key] = None

        return result

    def print_config(self, format_type: str = "tree"):
        """Print agent configuration in the specified format."""
        config_info = self.get_info()["config"]
        print(AgentPresenter.format_config(config_info, format_type))
        return

    def print_info(self, format_type: str = "tree", sections: Optional[List[str]] = None):
        """
        Print agent information in the specified format.

        Args:
            format_type: Presentation format ('table', 'tree', 'json')
            sections: Optional list of sections to display ('core', 'config', 'state')
                        If None, displays all sections
        """
        info = self.get_info()
        if sections:
            info = {k: v for k, v in info.items() if k in sections}
        print(AgentInfoPresenter.format_info(info, format_type))


class AgentPresenter:
    """Utility class for presenting agent information in various formats."""

    @staticmethod
    def format_config(config: Dict[str, Any], format_type: str = "table") -> str:
        """
        Format agent configuration in the specified format.

        Args:
            config: Dictionary containing agent configuration
            format_type: Presentation format ('table', 'tree', 'json')

        Returns:
            Formatted string representation of the configuration
        """
        if format_type == "table":
            return AgentPresenter._as_table(config)
        elif format_type == "tree":
            return AgentPresenter._as_tree(config)
        elif format_type == "json":
            return AgentPresenter._as_json(config)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    @staticmethod
    def _as_table(data: Dict[str, Any]) -> str:
        """Format data as a table using tabulate."""
        # Convert nested structures to string representation
        formatted_data = [[k, AgentPresenter._format_value(v)] for k, v in data.items()]
        return tabulate(formatted_data, headers=["Parameter", "Value"], tablefmt="grid")

    @staticmethod
    def _as_tree(data: Dict[str, Any]) -> str:
        """Format data as a tree structure using rich."""
        console = Console(record=True, stderr=False)  # Prevent double printing
        tree = Tree("Configuration")

        def add_to_tree(node: Tree, items: Dict[str, Any]):
            for key, value in items.items():
                if isinstance(value, dict):
                    branch = node.add(key)
                    add_to_tree(branch, value)
                else:
                    node.add(f"{key}: {AgentPresenter._format_value(value)}")

        add_to_tree(tree, data)
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    @staticmethod
    def _as_json(data: Dict[str, Any]) -> str:
        """Format data as formatted JSON."""
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, (int, float)):
            return f"{value:g}"  # Remove trailing zeros
        elif isinstance(value, np.ndarray):
            return np.array2string(value, precision=4, suppress_small=True)
        elif is_dataclass(value):
            return json.dumps(asdict(value), indent=2)
        return str(value)


class AgentInfoPresenter:
    """Utility class for presenting all agent information in various formats."""

    @staticmethod
    def format_info(info: Dict[str, Any], format_type: str = "table") -> str:
        """
        Format all agent information in the specified format.

        Args:
            info: Dictionary containing agent information (core, config, state)
            format_type: Presentation format ('table', 'tree', 'json')
        """
        if format_type == "table":
            return AgentInfoPresenter._as_table(info)
        elif format_type == "tree":
            return AgentInfoPresenter._as_tree(info)
        elif format_type == "json":
            return AgentInfoPresenter._as_json(info)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    @staticmethod
    def _as_table(data: Dict[str, Any]) -> str:
        """Format data as a set of tables using tabulate."""
        output = []

        for section, content in data.items():
            output.append(f"\n{section.upper()} INFORMATION:")
            if isinstance(content, dict):
                formatted_data = [
                    [k, AgentInfoPresenter._format_value(v)] for k, v in content.items()
                ]
                output.append(
                    tabulate(formatted_data, headers=["Parameter", "Value"], tablefmt="grid")
                )

        return "\n".join(output)

    @staticmethod
    def _as_tree(data: Dict[str, Any]) -> str:
        """Format data as a tree structure using rich."""
        console = Console(record=True, stderr=False)
        tree = Tree("Agent Information")

        def add_to_tree(node: Tree, items: Dict[str, Any]):
            for key, value in items.items():
                if isinstance(value, dict):
                    branch = node.add(f"[bold]{key}[/bold]")
                    add_to_tree(branch, value)
                else:
                    node.add(f"{key}: {AgentInfoPresenter._format_value(value)}")

        add_to_tree(tree, data)
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    @staticmethod
    def _as_json(data: Dict[str, Any]) -> str:
        """Format data as formatted JSON."""
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, (int, float)):
            return f"{value:g}"  # Remove trailing zeros
        elif isinstance(value, np.ndarray):
            return np.array2string(value, precision=4, suppress_small=True)
        elif is_dataclass(value):
            return json.dumps(asdict(value), indent=2)
        return str(value)
