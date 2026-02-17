
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


class FileManager:
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_path = Path(__file__).parent.parent.parent  # Project root
        self.config_path = config_path or self.base_path / "configs" / "config.example.json"
        self.config = self.load_configuration()
        self._setup_paths()
    
    def _setup_paths(self):
        self.paths = {
            'data': "data",
            'database_demo': "data/demo/database_demo.csv",
            'pareto_optimal': "data/pareto_optimal_points.csv",
            'figures': "figures"
        }
    
    def load_configuration(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def get_paths(self) -> Dict[str, str]:
        return self.paths
    
    def get_bounds(self) -> np.ndarray:
        return np.array([
            self.config['bounds']['lower'],
            self.config['bounds']['upper']
        ])
    
    def get_reference_point(self) -> np.ndarray:
        return np.array(self.config['reference_point'])


def get_file_manager(config_path: Optional[str] = None) -> FileManager:
    return FileManager(config_path)