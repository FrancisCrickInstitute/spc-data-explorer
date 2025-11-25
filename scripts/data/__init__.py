"""
Data handling package for the Phenotype Clustering Interactive Visualization App.

UPDATED (2025-01-21): Now supports loading BOTH SPC and CellProfiler datasets!

This package contains modules for loading, preprocessing, and managing data
for the phenotype clustering visualization application.

Modules:
- loader: Functions for loading CSV data and handling file operations
- landmark_loader: Functions for loading landmark analysis data (both SPC and CP)

The data package provides a clean interface for all data-related operations,
separating data handling logic from the UI and visualization components.
"""

try:
    from .loader import load_both_datasets, DataLoadError
except ImportError:
    from loader import load_both_datasets, DataLoadError

__all__ = ['load_both_datasets', 'DataLoadError']