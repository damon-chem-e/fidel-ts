#!/usr/bin/env python3
"""
Download Fidel-TS datasets from HuggingFace to the correct local directories.

This script downloads all datasets from the Fidel-TS collection on HuggingFace
and places them in the directory structure expected by the data configuration files.

Usage:
    python scripts/download_datasets.py [--datasets DATASET1,DATASET2] [--base_dir ./data]

Examples:
    # Download all datasets
    python scripts/download_datasets.py

    # Download specific datasets
    python scripts/download_datasets.py --datasets Bear_room,California_ISO

    # Specify custom base directory
    python scripts/download_datasets.py --base_dir /path/to/data
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    sys.exit(1)


# Mapping of local dataset names to HuggingFace repository IDs
# Based on: https://huggingface.co/collections/wxcai/fidel-ts
HF_DATASET_MAP: Dict[str, str] = {
    "Bear_room": "wxcai/Bear_room",
    "California_ISO": "wxcai/California_ISO",
    "Canada_photovoltaics_plants": "VEWOXIC/Canada_photovoltaics_plants",
    "Germany_Renewable_Power_Grid": "VEWOXIC/Germany_Renewable_Power_Grid",
    "Jena_Atmospheric_Physics": "VEWOXIC/Jena_Atmospheric_Physics",
    "NYC_traffic_speed": "VEWOXIC/NYC_traffic_speed",
}

# Mapping of dataset names to their target subdirectories based on data configs
# Some datasets have different subdirectory structures
DATASET_SUBDIRS: Dict[str, str] = {
    "Bear_room": "time_series",
    "California_ISO": "time_series",
    "Canada_photovoltaics_plants": "time_series",
    "Germany_Renewable_Power_Grid": "impute_data",  # Special case from fullGRPG.yaml
    "Jena_Atmospheric_Physics": "time_series",
    "NYC_traffic_speed": "time_series",
}


def download_dataset(
    dataset_name: str,
    hf_repo_id: str,
    target_dir: Path,
    subdir: str = "time_series",
) -> bool:
    """
    Download a single dataset from HuggingFace to the target directory.

    Args:
        dataset_name: Name of the dataset (for display)
        hf_repo_id: HuggingFace repository ID (e.g., "wxcai/Bear_room")
        target_dir: Base target directory (e.g., "./data")
        subdir: Subdirectory within the dataset folder (e.g., "time_series")

    Returns:
        True if download succeeded, False otherwise
    """
    # Construct full target path
    full_target_dir = target_dir / dataset_name / subdir
    full_target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"  HuggingFace: {hf_repo_id}")
    print(f"  Target: {full_target_dir}")
    print(f"{'='*60}")

    try:
        # Download the dataset
        # snapshot_download will create the directory if it doesn't exist
        downloaded_path = snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=str(full_target_dir),
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
        )

        print(f"✓ Successfully downloaded {dataset_name}")
        print(f"  Location: {downloaded_path}")
        return True

    except HfHubHTTPError as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        print(f"  Please check if the repository exists: https://huggingface.co/{hf_repo_id}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading {dataset_name}: {e}")
        return False


def main():
    """Main function to download datasets."""
    parser = argparse.ArgumentParser(
        description="Download Fidel-TS datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/download_datasets.py

  # Download specific datasets
  python scripts/download_datasets.py --datasets Bear_room,California_ISO

  # Use custom base directory
  python scripts/download_datasets.py --base_dir /path/to/data
        """,
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to download. "
        "Available: Bear_room, California_ISO, Canada_photovoltaics_plants, "
        "Germany_Renewable_Power_Grid, Jena_Atmospheric_Physics, NYC_traffic_speed. "
        "If not specified, all datasets will be downloaded.",
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="./data",
        help="Base directory for downloaded datasets (default: ./data)",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip datasets that already exist in the target directory",
    )

    args = parser.parse_args()

    # Determine which datasets to download
    if args.datasets:
        # Parse comma-separated list
        requested_datasets = [ds.strip() for ds in args.datasets.split(",")]
        # Validate dataset names
        invalid = [ds for ds in requested_datasets if ds not in HF_DATASET_MAP]
        if invalid:
            print(f"Error: Invalid dataset names: {', '.join(invalid)}")
            print(f"Available datasets: {', '.join(HF_DATASET_MAP.keys())}")
            sys.exit(1)
        datasets_to_download = requested_datasets
    else:
        # Download all datasets
        datasets_to_download = list(HF_DATASET_MAP.keys())

    # Convert base_dir to Path
    base_dir = Path(args.base_dir).resolve()

    print(f"\nFidel-TS Dataset Downloader")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    print(f"Datasets to download: {len(datasets_to_download)}")
    print(f"  {', '.join(datasets_to_download)}")
    print(f"{'='*60}")

    # Download each dataset
    results = {}
    for dataset_name in datasets_to_download:
        # Check if dataset already exists
        subdir = DATASET_SUBDIRS.get(dataset_name, "time_series")
        target_path = base_dir / dataset_name / subdir

        if args.skip_existing and target_path.exists() and any(target_path.iterdir()):
            print(f"\n⏭ Skipping {dataset_name} (already exists at {target_path})")
            results[dataset_name] = True
            continue

        hf_repo_id = HF_DATASET_MAP[dataset_name]
        success = download_dataset(
            dataset_name=dataset_name,
            hf_repo_id=hf_repo_id,
            target_dir=base_dir,
            subdir=subdir,
        )
        results[dataset_name] = success

    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    if successful:
        print(f"\n✓ Successfully downloaded ({len(successful)}):")
        for name in successful:
            subdir = DATASET_SUBDIRS.get(name, "time_series")
            print(f"  - {name} -> {base_dir / name / subdir}")

    if failed:
        print(f"\n✗ Failed to download ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")

    print(f"\n{'='*60}")

    # Exit with error code if any downloads failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

