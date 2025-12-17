"""
Helper script to create minimal id_info.json files for Time-MMD datasets.

Time-MMD datasets are single CSV files, so they need a minimal id_info.json
file to work with the Data_Provider structure.
"""

import json
import os
import argparse


def create_id_info_json(root_path, dataset_name=None):
    """
    Create a minimal id_info.json file for a Time-MMD dataset.
    
    Args:
        root_path: Root path of the dataset (where CSV file is located)
        dataset_name: Optional dataset name for description
    """
    # Minimal id_info structure for single-file Time-MMD datasets
    id_info = {
        'all': {
            'description': dataset_name if dataset_name else 'Time-MMD single-file dataset'
        }
    }
    
    # Create id_info.json in root_path
    id_info_path = os.path.join(root_path, 'id_info.json')
    
    with open(id_info_path, 'w') as f:
        json.dump(id_info, f, indent=2)
    
    print(f"✓ Created id_info.json at: {id_info_path}")
    print(f"  Content: {json.dumps(id_info, indent=2)}")
    return id_info_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create id_info.json for Time-MMD dataset")
    parser.add_argument(
        'root_path',
        type=str,
        help='Root path of the dataset (e.g., ./data/time_mmd/Traffic)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Optional dataset name for description'
    )
    
    args = parser.parse_args()
    
    create_id_info_json(args.root_path, args.name)

