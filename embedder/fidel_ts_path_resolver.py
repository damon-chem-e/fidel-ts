"""
Fidel-TS dataset-specific path resolver.

Hardcoded path resolution methods for each Fidel-TS subdataset to ensure robustness.
Since Fidel-TS is a static dataset, these methods are explicit and don't need to be extensible.
"""

from pathlib import Path
from typing import Dict, Any


class FidelTSPathResolver:
    """
    Hardcoded path resolver for Fidel-TS datasets.
    
    Each subdataset has its own specific method to ensure robustness.
    No generic/pattern-based resolution - everything is explicit and hardcoded.
    """
    
    # Dataset name mapping (extracted from root_path or config)
    DATASET_NAMES = [
        'Bear_room',
        'California_ISO',
        'Canada_photovoltaics_plants',
        'Germany_Renewable_Power_Grid',
        'Jena_Atmospheric_Physics',
        'NYC_traffic_speed'
    ]
    
    def __init__(self, dataset_name: str, hetero_info: Dict[str, Any], base_data_path: str):
        """
        Initialize path resolver for a Fidel-TS dataset.
        
        Args:
            dataset_name: Name of Fidel-TS dataset (must be in DATASET_NAMES)
            hetero_info: Config dict with root_path, formatter, static_path, etc.
            base_data_path: Base path to data directory (e.g., './data')
        
        Raises:
            ValueError: If dataset_name is not recognized
        """
        if dataset_name not in self.DATASET_NAMES:
            raise ValueError(
                f"Unknown Fidel-TS dataset: {dataset_name}. "
                f"Must be one of: {self.DATASET_NAMES}"
            )
        
        self.dataset_name = dataset_name
        self.hetero_info = hetero_info
        self.base_data_path = Path(base_data_path)
    
    def resolve_paths(self) -> Dict[str, Any]:
        """
        Main entry point - routes to subdataset-specific method.
        
        Returns:
            Dictionary with paths and dataset-specific information:
                - 'old_embedding_path' or 'old_embedding_paths': Path(s) to old embedding file(s)
                - 'old_text_path' or 'old_text_paths': Path(s) to raw text file(s) (if available)
                - 'old_static_path': Path to old static embeddings file (.pkl)
                - 'static_text_path': Path to static text JSON file (for re-embedding)
                - 'cache_base': Base path for new cache directory
                - Additional dataset-specific keys (years, source_type, etc.)
        
        Raises:
            NotImplementedError: If no resolver method exists for the dataset
        """
        # Convert dataset name to method name
        # Bear_room -> resolve_bear_room_paths
        # California_ISO -> resolve_california_iso_paths
        # etc.
        method_name = f'resolve_{self.dataset_name.lower().replace("-", "_").replace(" ", "_")}_paths'
        method = getattr(self, method_name, None)
        
        if method is None:
            raise NotImplementedError(
                f"No path resolver method for dataset: {self.dataset_name}. "
                f"Expected method: {method_name}"
            )
        
        result = method(self.hetero_info)
        
        # Add static_text_path (same for all datasets: base_data_path/dataset_name/static_info.json)
        result['static_text_path'] = self.base_data_path / self.dataset_name / "static_info.json"
        
        return result
    
    def resolve_bear_room_paths(self, hetero_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hardcoded path resolution for Bear_room dataset.
        
        Always uses wm_messages_v3.pkl (latest, hardcoded).
        Supports both weather and room sources (determined from root_path).
        """
        root_path = Path(hetero_info['root_path'])
        
        # Extract source type from root_path
        if 'weather' in str(root_path):
            source_type = 'weather'
            text_source_subdir = 'weather_report'
        elif 'room' in str(root_path):
            source_type = 'room'
            text_source_subdir = 'room_report'
        else:
            raise ValueError(
                f"Unknown source type in Bear_room root_path: {hetero_info['root_path']}. "
                f"Expected 'weather' or 'room' in path."
            )
        
        # Hardcoded: Always use v3 (latest)
        old_embedding_file = 'wm_messages_v3.pkl'
        old_text_file = 'wm_messages_v3.json'
        
        # Paths
        old_embedding_path = root_path / old_embedding_file
        old_text_path = root_path.parent.parent / text_source_subdir / 'formal_report' / old_text_file
        old_static_path = root_path / hetero_info['static_path']
        
        # New cache base path
        # From: data/Bear_room/hetero/weather/report_embedding/formal_report
        # To:   data/Bear_room/hetero/embeddings_cache/weather/report_embedding/formal_report
        cache_base = root_path.parent.parent.parent / 'embeddings_cache' / source_type / 'report_embedding' / 'formal_report'
        
        return {
            'old_embedding_path': old_embedding_path,
            'old_text_path': old_text_path,
            'old_static_path': old_static_path,
            'cache_base': cache_base,
            'source_type': source_type
        }
    
    def resolve_california_iso_paths(self, hetero_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hardcoded path resolution for California_ISO dataset.
        
        Loads ALL year files (2018-2025, hardcoded list) and merges them.
        """
        root_path = Path(hetero_info['root_path'])
        
        # Hardcoded: All years that should be loaded
        years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
        
        old_embedding_files = [f'fast_general_formal_embeddings_{year}.pkl' for year in years]
        old_embedding_paths = [root_path / f for f in old_embedding_files]
        old_static_path = root_path / hetero_info['static_path']
        
        # New cache base path
        # From: data/California_ISO/weather/merged_report_embedding
        # To:   data/California_ISO/weather/embeddings_cache/merged_report_embedding
        cache_base = root_path.parent / 'embeddings_cache' / 'merged_report_embedding'
        
        # Text source (if exists for re-embedding)
        text_source_base = root_path.parent / 'merged_general_report'
        old_text_paths = [text_source_base / 'merged_general_weather_forecast.json'] if text_source_base.exists() else []
        
        return {
            'old_embedding_paths': old_embedding_paths,  # List of paths
            'old_text_paths': old_text_paths,
            'old_static_path': old_static_path,
            'cache_base': cache_base,
            'years': years
        }
    
    def resolve_canada_photovoltaics_plants_paths(self, hetero_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hardcoded path resolution for Canada_photovoltaics_plants dataset.
        
        Loads ALL year files (2015-2023, hardcoded list) and merges them.
        """
        root_path = Path(hetero_info['root_path'])
        
        # Hardcoded: All years
        years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
        
        old_embedding_files = [f'fast_general_formal_embeddings_{year}.pkl' for year in years]
        old_embedding_paths = [root_path / f for f in old_embedding_files]
        old_static_path = root_path / hetero_info['static_path']
        
        # New cache base path
        # From: data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report
        # To:   data/Canada_photovoltaics_plants/weather/embeddings_cache/calgary/report_embedding/formal_report
        cache_base = root_path.parent.parent.parent / 'embeddings_cache' / 'calgary' / 'report_embedding' / 'formal_report'
        
        return {
            'old_embedding_paths': old_embedding_paths,
            'old_text_paths': [],  # Text source TBD (may not exist)
            'old_static_path': old_static_path,
            'cache_base': cache_base,
            'years': years
        }
    
    def resolve_germany_renewable_power_grid_paths(self, hetero_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hardcoded path resolution for Germany_Renewable_Power_Grid dataset.
        
        Loads ALL year files (2011-2021, hardcoded list) and merges them.
        """
        root_path = Path(hetero_info['root_path'])
        
        # Hardcoded: All years (based on dataset time range)
        years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        
        old_embedding_files = [f'fast_general_formal_embeddings_{year}.pkl' for year in years]
        old_embedding_paths = [root_path / f for f in old_embedding_files]
        old_static_path = root_path / hetero_info['static_path']
        
        # New cache base path
        # From: data/Germany_Renewable_Power_Grid/weather/merged_report_embedding
        # To:   data/Germany_Renewable_Power_Grid/weather/embeddings_cache/merged_report_embedding
        cache_base = root_path.parent / 'embeddings_cache' / 'merged_report_embedding'
        
        return {
            'old_embedding_paths': old_embedding_paths,
            'old_text_paths': [],  # Text source TBD
            'old_static_path': old_static_path,
            'cache_base': cache_base,
            'years': years
        }
    
    def resolve_jena_atmospheric_physics_paths(self, hetero_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hardcoded path resolution for Jena_Atmospheric_Physics dataset.
        
        Always uses wm_messages_v3.pkl (latest, hardcoded).
        """
        root_path = Path(hetero_info['root_path'])
        
        # Hardcoded: Always use v3 (latest)
        old_embedding_file = 'wm_messages_v3.pkl'
        old_text_file = 'wm_messages_v3.json'
        
        old_embedding_path = root_path / old_embedding_file
        old_static_path = root_path / hetero_info['static_path']
        
        # Text source
        old_text_path = root_path.parent.parent / 'weather_report' / 'formal_report' / old_text_file
        
        # New cache base path
        # From: data/Jena_Atmospheric_Physics/weather/report_embedding/formal_report
        # To:   data/Jena_Atmospheric_Physics/weather/embeddings_cache/report_embedding/formal_report
        cache_base = root_path.parent.parent / 'embeddings_cache' / 'report_embedding' / 'formal_report'
        
        return {
            'old_embedding_path': old_embedding_path,
            'old_text_path': old_text_path,
            'old_static_path': old_static_path,
            'cache_base': cache_base
        }
    
    def resolve_nyc_traffic_speed_paths(self, hetero_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hardcoded path resolution for NYC_traffic_speed dataset.
        
        Loads ALL year files (2018-2022, hardcoded list) and merges them.
        """
        root_path = Path(hetero_info['root_path'])
        
        # Hardcoded: All years (based on dataset time range)
        years = ['2018', '2019', '2020', '2021', '2022']
        
        old_embedding_files = [f'fast_general_formal_embeddings_{year}.pkl' for year in years]
        old_embedding_paths = [root_path / f for f in old_embedding_files]
        old_static_path = root_path / hetero_info['static_path']
        
        # New cache base path
        # From: data/NYC_traffic_speed/weather/merged_report_embedding
        # To:   data/NYC_traffic_speed/weather/embeddings_cache/merged_report_embedding
        cache_base = root_path.parent / 'embeddings_cache' / 'merged_report_embedding'
        
        return {
            'old_embedding_paths': old_embedding_paths,
            'old_text_paths': [],  # Text source TBD
            'old_static_path': old_static_path,
            'cache_base': cache_base,
            'years': years
        }

