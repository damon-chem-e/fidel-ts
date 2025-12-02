#!/usr/bin/env python3
"""
Map data directory structure to JSON.

Usage:
    python scripts/map_data_structure.py --data-dir /path/to/data --output data_structure.json
"""

import json
import os
from pathlib import Path
import argparse
from typing import Dict, Any

def map_directory(root_dir: Path, max_depth: int = 20) -> Dict[str, Any]:
    """
    Recursively map directory structure to nested dictionary.
    
    Returns structure like:
    {
        "type": "directory",
        "name": "data",
        "path": "data",
        "children": {
            "dataset1": {
                "type": "directory",
                "files": [...],
                "children": {...}
            }
        }
    }
    """
    def _map_path(path: Path, current_depth: int = 0) -> Dict[str, Any]:
        if current_depth > max_depth:
            return {"type": "truncated", "path": str(path)}
        
        result = {
            "type": "directory" if path.is_dir() else "file",
            "name": path.name,
            "path": str(path.relative_to(root_dir))
        }
        
        if path.is_dir():
            children = {}
            files = []
            
            try:
                for item in sorted(path.iterdir()):
                    if item.is_file():
                        file_info = {
                            "name": item.name,
                            "path": str(item.relative_to(root_dir)),
                            "size": item.stat().st_size,
                            "extension": item.suffix
                        }
                        files.append(file_info)
                    elif item.is_dir():
                        children[item.name] = _map_path(item, current_depth + 1)
            except PermissionError:
                result["error"] = "Permission denied"
                return result
            
            if children:
                result["children"] = children
            if files:
                result["files"] = files
        
        return result
    
    root_info = _map_path(root_dir)
    return root_info

def main():
    parser = argparse.ArgumentParser(
        description="Map directory structure to JSON"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_structure.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory does not exist: {data_dir}")
        return 1
    
    print(f"Mapping directory structure: {data_dir}")
    print("This may take a while for large directories...")
    
    structure = map_directory(data_dir)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(structure, f, indent=2)
    
    print(f"✓ Structure mapped to: {output_path}")
    
    # Print summary
    def count_files(node, count=0):
        if isinstance(node, dict):
            if node.get("type") == "file":
                return count + 1
            elif "files" in node:
                count += len(node["files"])
            if "children" in node:
                for child in node["children"].values():
                    count = count_files(child, count)
        return count
    
    total_files = count_files(structure)
    print(f"✓ Total files found: {total_files}")
    
    return 0

if __name__ == "__main__":
    exit(main())

