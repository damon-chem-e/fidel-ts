#!/usr/bin/env python3
"""
Diagnostic script to inspect embedding cache files.

Usage:
    python scripts/inspect_embedding_cache.py data/time_mmd/Traffic/embeddings_3af7b92ef045b153
"""

import sys
import joblib
import pickle
from pathlib import Path
import json

def inspect_cache(cache_dir: Path):
    """Inspect an embedding cache directory."""
    print(f"Inspecting cache directory: {cache_dir}")
    print("=" * 70)
    
    if not cache_dir.exists():
        print(f"ERROR: Directory does not exist: {cache_dir}")
        return
    
    # Check metadata.json
    metadata_path = cache_dir / 'metadata.json'
    if metadata_path.exists():
        print("\n[Metadata]")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(json.dumps(metadata, indent=2))
        except Exception as e:
            print(f"ERROR loading metadata: {e}")
    else:
        print("\n[Metadata] NOT FOUND")
    
    # Try to load embeddings.pkl
    emb_path = cache_dir / 'embeddings.pkl'
    if not emb_path.exists():
        print(f"\n[Embeddings] File not found: {emb_path}")
        return
    
    print(f"\n[Embeddings] File: {emb_path}")
    print(f"  Size: {emb_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Try joblib.load first (standard format)
    print("\n  Attempting joblib.load()...")
    try:
        embeddings = joblib.load(emb_path)
        print(f"  ✓ Successfully loaded with joblib")
        print(f"  Type: {type(embeddings)}")
        
        if isinstance(embeddings, dict):
            print(f"  Keys: {len(embeddings)} entries")
            if len(embeddings) > 0:
                sample_key = next(iter(embeddings.keys()))
                sample_value = embeddings[sample_key]
                print(f"  Sample key: {repr(sample_key)} (type: {type(sample_key).__name__})")
                print(f"  Sample value type: {type(sample_value)}")
                if hasattr(sample_value, 'shape'):
                    print(f"  Sample value shape: {sample_value.shape}")
                if hasattr(sample_value, 'dtype'):
                    print(f"  Sample value dtype: {sample_value.dtype}")
                
                # Show first few keys
                print(f"\n  First 10 keys:")
                for i, key in enumerate(list(embeddings.keys())[:10]):
                    print(f"    [{i}] {repr(key)}")
                
                # Check if keys match expected format
                print(f"\n  Key format analysis:")
                key_types = {}
                for key in embeddings.keys():
                    key_type = type(key).__name__
                    key_types[key_type] = key_types.get(key_type, 0) + 1
                for ktype, count in key_types.items():
                    print(f"    {ktype}: {count} keys")
                
                # Check for specific timestamp format
                expected_keys = ['19800101000000', '19800201000000', '19800301000000']
                print(f"\n  Checking for expected keys:")
                for exp_key in expected_keys:
                    found = exp_key in embeddings
                    found_int = int(exp_key) in embeddings if exp_key.isdigit() else False
                    print(f"    '{exp_key}': {found} (as string)")
                    if exp_key.isdigit():
                        print(f"    {int(exp_key)}: {found_int} (as int)")
        else:
            print(f"  WARNING: Expected dict, got {type(embeddings)}")
            print(f"  Contents: {embeddings}")
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        
        # Try pickle.load as fallback
        print("\n  Attempting pickle.load()...")
        try:
            with open(emb_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"  ✓ Successfully loaded with pickle")
            print(f"  Type: {type(embeddings)}")
            if isinstance(embeddings, dict):
                print(f"  Keys: {len(embeddings)} entries")
        except Exception as e2:
            print(f"  ✗ Also failed: {e2}")
            print(f"  Error type: {type(e2).__name__}")
            
            # Try to read raw bytes
            print("\n  Attempting to read raw file header...")
            try:
                with open(emb_path, 'rb') as f:
                    header = f.read(100)
                print(f"  First 100 bytes (hex): {header.hex()}")
                print(f"  First 100 bytes (repr): {repr(header)}")
            except Exception as e3:
                print(f"  ✗ Failed to read file: {e3}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_embedding_cache.py <cache_dir>")
        print("Example: python scripts/inspect_embedding_cache.py data/time_mmd/Traffic/embeddings_3af7b92ef045b153")
        sys.exit(1)
    
    cache_dir = Path(sys.argv[1])
    inspect_cache(cache_dir)
