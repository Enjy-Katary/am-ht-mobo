import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional


class NotebookFileManager:
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_path = Path(__file__).parent.parent 
        self.config_path = config_path or self.base_path / "configs" / "config.example.json"
        self.config = self.load_configuration()
        self._setup_paths()
    
    def _setup_paths(self):
        self.paths = {
            'database_demo': "mobo_data.csv",  
            'figures': "."  
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

def get_notebook_file_manager(config_path: Optional[str] = None) -> NotebookFileManager:
    return NotebookFileManager(config_path)


def patch_bo_module_for_notebook():
    import sys
    sys.path.append('../src')
    from am_ht_mobo import bo
    
    notebook_fm = get_notebook_file_manager()
    
    bo.config = notebook_fm.config
    bo.FIGURES_PATH = notebook_fm.paths['figures']
    bo.DATABASE_PATH = notebook_fm.paths['database_demo']  
    bo.BOUNDS = notebook_fm.get_bounds()
    bo.REFERENCE = notebook_fm.get_reference_point()
    bo.CSV_PATH = "notebook_pareto_points.csv"
    bo.DATA_PATH = None
    
    return bo

def create_data_loader(database_path: str):
    def load_data():
        df = pl.read_csv(database_path)
        
        points = df.select(["T1 (C)", "t1 (min)", "T2 (C)", "t2 (min)"]).to_numpy()
        results_original = df.select(["Ultimate Tensile Strength (MPa)", "Energy (kWh)"]).to_numpy()
        
        results_transformed = np.copy(results_original)
        results_transformed[:, 1] = 1.0 / results_original[:, 1]
        
        return points, results_transformed, results_original
    
    return load_data


def patch_plotting_module_for_notebook():
    import sys
    sys.path.append('../src')
    from am_ht_mobo import plotting
    
    notebook_fm = get_notebook_file_manager()
    plotting.METRIC_FIGURES_PATH = notebook_fm.paths['figures']
    plotting.load_data_from_database = create_data_loader(notebook_fm.paths['database_demo'])
    
    return plotting


def setup_notebook_environment():
    bo_module = patch_bo_module_for_notebook()
    plotting_module = patch_plotting_module_for_notebook()
    
    return bo_module, plotting_module


def calculate_hypervolume_progress(bo_module, train_x, train_y_transformed, train_y_original, config):
    n_initial = len(config['initial_process_parameters']['T1 (C)'])
    hypervolumes = []
    iterations_list = []
    
    for i in range(n_initial, len(train_x) + 1):
        subset_x = train_x[:i]
        subset_y_transformed = train_y_transformed[:i]
        subset_y_original = train_y_original[:i]
        
        pareto_x_subset, pareto_y_subset, _ = bo_module.identify_pareto_front(
            subset_x, subset_y_transformed, subset_y_original
        )
        hv = bo_module.calculate_hypervolume(pareto_y_subset)
        hypervolumes.append(hv)
        iterations_list.append(i - n_initial + 1)
    
    return np.array(iterations_list), np.array(hypervolumes)