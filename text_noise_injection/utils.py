import hashlib
import json
import os
from datetime import datetime
from .config import InjectionConfig

def generate_dataset_id(config: InjectionConfig) -> str:
    """
    Generates a unique ID for the dataset based on the current date and a hash of the configuration.
    Format: YYYYMMDD_{hash}
    """
    # Create a consistent string representation of the config
    config_str = json.dumps(config.to_dict(), sort_keys=True)
    
    # Generate MD5 hash (short enough for directory names)
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
    
    # Get current date
    date_str = datetime.now().strftime("%Y%m%d")
    
    return f"{date_str}_{config_hash}"

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

