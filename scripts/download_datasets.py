#!/usr/bin/env python3
"""
Download Fidel-TS datasets from HuggingFace to the correct local directories.

This script downloads all datasets from the Fidel-TS collection on HuggingFace
and places them in the directory structure expected by the data configuration files.

Usage:
    python scripts/download_datasets.py [--datasets DATASET1,DATASET2] [--base-dir ./data]

Examples:
    # Download all datasets
    python scripts/download_datasets.py

    # Download specific datasets
    python scripts/download_datasets.py --datasets Bear_room,California_ISO

    # Specify custom base directory
    python scripts/download_datasets.py --base-dir /path/to/data
"""

import sys
from pathlib import Path
from typing import Dict, Optional

try:
    import typer
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    missing_package = "typer" if "typer" in str(e) else "huggingface_hub"
    print(f"Error: {missing_package} is not installed.")
    print(f"Please install it with: pip install {missing_package}")
    sys.exit(1)

# Create Typer app
app = typer.Typer(
    name="download-datasets",
    help="Download Fidel-TS datasets from HuggingFace to the correct local directories.",
    add_completion=False,
)


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

# Mapping of dataset names to their time series subdirectories (for checking if data exists)
# This is where the time series data should be after download
DATASET_TIME_SERIES_DIRS: Dict[str, str] = {
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
) -> bool:
    """
    Download a single dataset from HuggingFace to the target directory.
    
    Downloads to the dataset root directory to preserve the repository structure
    (which includes time_series, weather, hetero, etc. at the same level).

    Args:
        dataset_name: Name of the dataset (for display)
        hf_repo_id: HuggingFace repository ID (e.g., "wxcai/Bear_room")
        target_dir: Base target directory (e.g., "./data")

    Returns:
        True if download succeeded, False otherwise
    """
    # Download to dataset root directory to preserve HuggingFace repo structure
    # The repo structure has time_series, weather, hetero, etc. at the same level
    full_target_dir = target_dir / dataset_name
    full_target_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Downloading: {dataset_name}")
    typer.echo(f"  HuggingFace: {hf_repo_id}")
    typer.echo(f"  Target: {full_target_dir}")
    typer.echo(f"{'='*60}")

    try:
        # Download the dataset
        # snapshot_download will create the directory if it doesn't exist
        downloaded_path = snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=str(full_target_dir),
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
        )

        typer.echo(f"✓ Successfully downloaded {dataset_name}")
        typer.echo(f"  Location: {downloaded_path}")
        return True

    except HfHubHTTPError as e:
        typer.echo(f"✗ Error downloading {dataset_name}: {e}", err=True)
        typer.echo(
            f"  Please check if the repository exists: https://huggingface.co/{hf_repo_id}",
            err=True,
        )
        return False
    except Exception as e:
        typer.echo(f"✗ Unexpected error downloading {dataset_name}: {e}", err=True)
        return False


@app.command()
def main(
    datasets: Optional[str] = typer.Option(
        None,
        "--datasets",
        "-d",
        help="Comma-separated list of datasets to download. "
        "Available: Bear_room, California_ISO, Canada_photovoltaics_plants, "
        "Germany_Renewable_Power_Grid, Jena_Atmospheric_Physics, NYC_traffic_speed. "
        "If not specified, all datasets will be downloaded.",
    ),
    base_dir: Path = typer.Option(
        Path("./data"),
        "--base-dir",
        "-b",
        help="Base directory for downloaded datasets (default: ./data)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force overwrite of existing datasets. By default, existing datasets are skipped.",
    ),
) -> None:
    """
    Download Fidel-TS datasets from HuggingFace.

    By default, existing datasets are skipped. Use --force to overwrite them.

    Examples:
        # Download all datasets (skip existing)
        python scripts/download_datasets.py

        # Download specific datasets
        python scripts/download_datasets.py --datasets Bear_room,California_ISO

        # Use custom base directory
        python scripts/download_datasets.py --base-dir /path/to/data

        # Force overwrite existing datasets
        python scripts/download_datasets.py --force
    """
    # Determine which datasets to download
    if datasets:
        # Parse comma-separated list
        requested_datasets = [ds.strip() for ds in datasets.split(",")]
        # Validate dataset names
        invalid = [ds for ds in requested_datasets if ds not in HF_DATASET_MAP]
        if invalid:
            typer.echo(
                f"Error: Invalid dataset names: {', '.join(invalid)}",
                err=True,
            )
            typer.echo(
                f"Available datasets: {', '.join(HF_DATASET_MAP.keys())}",
                err=True,
            )
            raise typer.Exit(code=1)
        datasets_to_download = requested_datasets
    else:
        # Download all datasets
        datasets_to_download = list(HF_DATASET_MAP.keys())

    # Convert base_dir to Path and resolve
    base_dir = base_dir.resolve()

    typer.echo("\nFidel-TS Dataset Downloader")
    typer.echo("=" * 60)
    typer.echo(f"Base directory: {base_dir}")
    typer.echo(f"Force overwrite: {force}")
    typer.echo(f"Datasets to process: {len(datasets_to_download)}")
    typer.echo(f"  {', '.join(datasets_to_download)}")
    typer.echo("=" * 60)

    # Download each dataset
    results: Dict[str, bool] = {}
    skipped: Dict[str, Path] = {}
    for dataset_name in datasets_to_download:
        # Check if dataset already exists by looking for time series directory
        time_series_subdir = DATASET_TIME_SERIES_DIRS.get(dataset_name, "time_series")
        target_path = base_dir / dataset_name / time_series_subdir

        # Skip existing datasets unless force is enabled
        if not force and target_path.exists() and any(target_path.iterdir()):
            typer.echo(f"\n⏭ Skipping {dataset_name}")
            typer.echo(f"   Dataset already exists at: {target_path}")
            typer.echo("   Use --force to overwrite")
            skipped[dataset_name] = target_path
            results[dataset_name] = True
            continue

        hf_repo_id = HF_DATASET_MAP[dataset_name]
        success = download_dataset(
            dataset_name=dataset_name,
            hf_repo_id=hf_repo_id,
            target_dir=base_dir,
        )
        results[dataset_name] = success

    # Print summary
    typer.echo("\n" + "=" * 60)
    typer.echo("Download Summary")
    typer.echo("=" * 60)
    successful = [name for name, success in results.items() if success and name not in skipped]
    failed = [name for name, success in results.items() if not success]

    if skipped:
        typer.echo(f"\n⏭ Skipped (already exist) ({len(skipped)}):")
        for name, path in skipped.items():
            typer.echo(f"  - {name} -> {path}")

    if successful:
        typer.echo(f"\n✓ Successfully downloaded ({len(successful)}):")
        for name in successful:
            time_series_subdir = DATASET_TIME_SERIES_DIRS.get(name, "time_series")
            typer.echo(f"  - {name} -> {base_dir / name} (time series: {base_dir / name / time_series_subdir})")

    if failed:
        typer.echo(f"\n✗ Failed to download ({len(failed)}):")
        for name in failed:
            typer.echo(f"  - {name}")

    typer.echo("\n" + "=" * 60)

    # Exit with error code if any downloads failed
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

