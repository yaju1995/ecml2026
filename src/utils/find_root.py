

"""
find_project_root.py

Utility function to find the root directory of a project by searching for specific marker files.

Author: Eloann Le Guern--Dall'o
License: Apache-2.0
Copyright 2025 Eloann Le Guern--Dall'o
See LICENSE for details.
"""

from pathlib import Path

def find_project_root(marker_files=None):
    """
    Find project root by looking for marker files
    Default markers: pyproject.toml, setup.py, .git, requirements.txt
    """
    if marker_files is None:
        marker_files = ['pyproject.toml']
    
    current_path = Path.cwd()
    
    # Search up the directory tree
    for parent in [current_path] + list(current_path.parents):
        for marker in marker_files:
            if (parent / marker).exists():
                return parent
    
    # If no marker found, return current directory
    return current_path