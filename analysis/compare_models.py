
import typer
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List

app = typer.Typer()

# ============================================================================
# Helper Functions
# ============================================================================

def extract_entity_id(sample_id: str) -> str:
    """Extract entity_id from sample_id: 'entity_id|timestamp|sequence_index'"""
    return sample_id.split('|')[0]

def extract_timestamp(sample_id: str) -> str:
    """Extract timestamp from sample_id"""
    return sample_id.split('|')[1]

def extract_sequence_index(sample_id: str) -> int:
    """Extract sequence_index from sample_id"""
    return int(sample_id.split('|')[2])

def detect_data_level(df: pl.DataFrame) -> str:
    """
    Detect if DataFrame contains per-channel data.
    Returns: "channel" if channel_id is not null for any rows, else "sample"
    """
    if "channel_id" in df.columns:
        has_channels = df.filter(pl.col("channel_id").is_not_null()).height > 0
        return "channel" if has_channels else "sample"
    return "sample"

def aggregate_to_sample_level(df: pl.DataFrame) -> pl.DataFrame:
    """
    If DataFrame has per-channel data, aggregate to sample level.
    Uses loss_sample_avg if available, otherwise computes mean of channel losses.
    """
    if "channel_id" not in df.columns:
        return df
    
    # Check if we have per-channel data
    has_channels = df.filter(pl.col("channel_id").is_not_null()).height > 0
    if not has_channels:
        return df
    
    # Group by sample-level columns and aggregate
    group_cols = ["sample_id", "epoch", "split", "entity_id", "timestamp"]
    group_cols = [col for col in group_cols if col in df.columns]
    
    # Use loss_sample_avg if available, otherwise compute mean
    if "loss_sample_avg" in df.columns:
        # Filter to rows where loss_sample_avg is not null (sample-level rows)
        sample_rows = df.filter(pl.col("loss_sample_avg").is_not_null())
        if sample_rows.height > 0:
            return sample_rows.select([col for col in sample_rows.columns if col != "channel_id"])
    
    # Otherwise, compute mean of channel losses per sample
    aggregated = df.filter(pl.col("channel_id").is_not_null()).group_by(group_cols).agg([
        pl.col("loss").mean().alias("loss"),
        pl.col("loss_sample_avg").first().alias("loss_sample_avg")  # Keep if exists
    ])
    
    return aggregated

def filter_channels(df: pl.DataFrame, channel_list: List[str]) -> pl.DataFrame:
    """Filter DataFrame to specific channels."""
    if "channel_id" not in df.columns:
        return df
    
    return df.filter(pl.col("channel_id").is_in(channel_list))

def ensure_sample_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure sample_id column exists. If missing, generate from entity_id and timestamp.
    This provides backward compatibility with old data format.
    """
    if "sample_id" in df.columns:
        return df
    
    # Generate sample_id from entity_id and timestamp (with sequence_index=0)
    if "entity_id" in df.columns and "timestamp" in df.columns:
        df = df.with_columns(
            (pl.col("entity_id").fill_null("unknown") + "|" + 
             pl.col("timestamp").cast(pl.Utf8) + "|0").alias("sample_id")
        )
    else:
        raise ValueError("Cannot generate sample_id: missing entity_id or timestamp columns")
    
    return df

# ============================================================================
# Data Loading
# ============================================================================

def _load_epoch_file(per_sample_dir: Path, epoch: Optional[int], split_filter: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Load data from epoch files, optionally filtering by split.
    
    Args:
        per_sample_dir: Directory containing per-sample metrics
        epoch: Specific epoch to load. If None, loads the latest available epoch.
        split_filter: Optional split name to filter ('train', 'val', 'test'). If None, returns all splits.
    
    Returns:
        DataFrame with metrics, optionally filtered by split
    """
    files = sorted(per_sample_dir.glob("epoch_*.parquet"))
    if not files:
        return None
    
    if epoch is None:
        target_file = files[-1]
        print(f"Loading metrics from latest epoch: {target_file.name}")
    else:
        target_file = per_sample_dir / f"epoch_{epoch:03d}.parquet"
        if not target_file.exists():
            raise ValueError(f"Metrics for epoch {epoch} not found")
        print(f"Loading metrics from epoch: {target_file.name}")
    
    df = pl.read_parquet(target_file)
    
    # Filter by split if requested
    if split_filter is not None:
        if "split" not in df.columns:
            raise ValueError(f"Split column not found in metrics file. Cannot filter by split '{split_filter}'")
        df = df.filter(pl.col("split") == split_filter)
        if df.height == 0:
            print(f"Warning: No {split_filter} data found in {target_file.name}")
            return None
    
    return df

def _load_split_file(per_sample_dir: Path, split_name: str, epoch: Optional[int] = None) -> Optional[pl.DataFrame]:
    """
    Load a single split from epoch files (backward compatibility function).
    
    In the new format, val and test splits are stored in epoch files, not separate files.
    This function loads from epoch files and filters by split.
    """
    return _load_epoch_file(per_sample_dir, epoch, split_filter=split_name)

def load_metrics(
    experiment_dir: str, 
    split: Optional[str] = None,  # 'train', 'val', 'test', or None for all
    epoch: Optional[int] = None   # Epoch to load (applies to all splits in new format)
) -> pl.DataFrame:
    """
    Load per-sample metrics from an experiment directory.
    
    In the new format, all splits (train/val/test) are stored together in epoch files.
    Each epoch file contains rows with a 'split' column indicating the dataset split.
    
    Args:
        experiment_dir: Path to experiment output directory.
        split: Dataset split to load ('train', 'val', 'test'). If None, loads all available splits.
        epoch: Specific epoch to load. If None, loads the last available epoch.
    
    Returns:
        DataFrame with per-sample metrics, optionally filtered by split
    """
    # New location: experiment_dir/metrics/per_sample/
    per_sample_dir = Path(experiment_dir) / "metrics" / "per_sample"
    
    # Fallback to old location for backward compatibility
    if not per_sample_dir.exists():
        old_per_sample_dir = Path(experiment_dir) / "checkpoints" / "per_sample"
        if old_per_sample_dir.exists():
            print(f"Warning: Using legacy location {old_per_sample_dir}. New location is {per_sample_dir}")
            per_sample_dir = old_per_sample_dir
        else:
            # Try even older location
            old_per_sample_dir = Path(experiment_dir) / "per_sample"
            if old_per_sample_dir.exists():
                print(f"Warning: Using legacy location {old_per_sample_dir}. New location is {per_sample_dir}")
                per_sample_dir = old_per_sample_dir
            else:
                raise ValueError(f"No per-sample metrics found in {experiment_dir}. Expected location: {per_sample_dir}")
    
    # Load from epoch files (new format: all splits in epoch files)
    df = _load_epoch_file(per_sample_dir, epoch, split_filter=split)
    
    if df is None or df.height == 0:
        raise ValueError(f"No metrics found in {per_sample_dir} for split={split}, epoch={epoch}")
    
    # Ensure sample_id exists (backward compatibility)
    df = ensure_sample_id(df)
    
    return df

# ============================================================================
# Comparison Logic
# ============================================================================

def _determine_comparison_level(
    baseline_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    level: str
) -> tuple[str, pl.DataFrame, pl.DataFrame]:
    """
    Determine the comparison level and prepare dataframes accordingly.
    
    Returns:
        Tuple of (level, prepared_baseline_df, prepared_candidate_df)
    """
    if level == "auto":
        baseline_level = detect_data_level(baseline_df)
        candidate_level = detect_data_level(candidate_df)
        
        # If both have channels, do channel-level comparison
        # If one has channels and one doesn't, aggregate to sample level
        if baseline_level == "channel" and candidate_level == "channel":
            level = "channel"
        else:
            # Aggregate both to sample level
            baseline_df = aggregate_to_sample_level(baseline_df)
            candidate_df = aggregate_to_sample_level(candidate_df)
            level = "sample"
    
    return level, baseline_df, candidate_df

def _determine_join_keys(
    baseline_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    level: str
) -> tuple[str, list[str], pl.DataFrame, pl.DataFrame]:
    """
    Determine join keys based on comparison level.
    
    Returns:
        Tuple of (final_level, join_keys, prepared_baseline_df, prepared_candidate_df)
    """
    if level == "channel":
        # Join on sample_id, channel_id, and split
        join_keys = ["sample_id", "channel_id", "split"]
        # Fallback if channel_id missing
        if "channel_id" not in baseline_df.columns or "channel_id" not in candidate_df.columns:
            level = "sample"
            baseline_df = aggregate_to_sample_level(baseline_df)
            candidate_df = aggregate_to_sample_level(candidate_df)
            join_keys = ["sample_id", "split"]
    else:
        # Sample-level join
        join_keys = ["sample_id", "split"]
    
    return level, join_keys, baseline_df, candidate_df

def _perform_join(
    baseline_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    join_keys: list[str]
) -> pl.DataFrame:
    """
    Perform the join operation with fallback logic.
    """
    # Ensure join keys exist
    for key in join_keys:
        if key not in baseline_df.columns:
            raise ValueError(f"Missing required column '{key}' in baseline data")
        if key not in candidate_df.columns:
            raise ValueError(f"Missing required column '{key}' in candidate data")
    
    # Join on sample_id (or fallback to entity_id + timestamp)
    try:
        joined = baseline_df.join(
            candidate_df,
            on=join_keys,
            suffix="_candidate",
            how="inner"
        )
    except Exception:
        # Fallback: try with entity_id + timestamp if sample_id join fails
        if "sample_id" in join_keys:
            fallback_keys = [k for k in ["entity_id", "timestamp", "split"] if k in baseline_df.columns]
            if len(fallback_keys) >= 2:
                print("Warning: Falling back to entity_id + timestamp join (sample_id may be inconsistent)")
                joined = baseline_df.join(
                    candidate_df,
                    on=fallback_keys,
                    suffix="_candidate",
                    how="inner"
                )
            else:
                raise
        else:
            raise
    
    return joined

def _compute_loss_differences(joined: pl.DataFrame) -> pl.DataFrame:
    """
    Compute loss differences between candidate and baseline.
    """
    # Rename loss columns
    if "loss" in joined.columns:
        joined = joined.rename({"loss": "loss_baseline"})
    
    # Compute difference: candidate - baseline
    # Negative values mean candidate is better (lower loss)
    if "loss_candidate" in joined.columns and "loss_baseline" in joined.columns:
        joined = joined.with_columns(
            (pl.col("loss_candidate") - pl.col("loss_baseline")).alias("diff")
        )
    else:
        raise ValueError("Could not find loss columns in joined data")
    
    return joined

def compute_comparison(
    baseline_df: pl.DataFrame, 
    candidate_df: pl.DataFrame,
    level: str = "auto"  # "sample", "channel", or "auto" (detect from data)
) -> pl.DataFrame:
    """
    Compare two dataframes by joining on sample_id (or fallback to entity_id + timestamp).
    
    Args:
        baseline_df: Baseline experiment metrics
        candidate_df: Candidate experiment metrics
        level: Comparison level - "sample", "channel", or "auto" (detect from data)
    
    Returns:
        Joined DataFrame with loss differences
    """
    # Determine comparison level and prepare dataframes
    level, baseline_df, candidate_df = _determine_comparison_level(baseline_df, candidate_df, level)
    
    # Determine join keys
    level, join_keys, baseline_df, candidate_df = _determine_join_keys(baseline_df, candidate_df, level)
    
    # Perform join
    joined = _perform_join(baseline_df, candidate_df, join_keys)
    
    # Compute differences
    joined = _compute_loss_differences(joined)
    
    return joined

# ============================================================================
# Statistics Computation
# ============================================================================

def compute_channel_stats(comparison_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-channel comparison statistics.
    Groups by split and channel_id to show which channels benefit most.
    """
    if "channel_id" not in comparison_df.columns:
        return pl.DataFrame()
    
    # Filter to rows with channel_id
    channel_data = comparison_df.filter(pl.col("channel_id").is_not_null())
    if channel_data.height == 0:
        return pl.DataFrame()
    
    stats = channel_data.group_by(["split", "channel_id"]).agg([
        pl.col("diff").mean().alias("mean_diff"),
        pl.col("diff").median().alias("median_diff"),
        pl.col("diff").std().alias("std_diff"),
        (pl.col("diff") < 0).mean().alias("improvement_rate"),
        pl.col("diff").count().alias("count")
    ]).sort("mean_diff")
    
    return stats

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_sample_distributions(comparison_df: pl.DataFrame, output_dir: Path):
    """Plot sample-level distribution of differences."""
    # Aggregate to sample level if needed
    sample_df = aggregate_to_sample_level(comparison_df)
    
    # Distribution of differences
    plt.figure(figsize=(10, 6))
    for split in sample_df["split"].unique():
        split_data = sample_df.filter(pl.col("split") == split)
        sns.histplot(split_data["diff"].to_numpy(), label=split, alpha=0.5, kde=True)
    
    plt.title("Distribution of Loss Differences (Candidate - Baseline)")
    plt.xlabel("Loss Difference (Negative = Candidate Better)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "diff_distribution.png", dpi=150)
    plt.close()

def plot_channel_distributions(comparison_df: pl.DataFrame, output_dir: Path):
    """
    Create visualizations for per-channel loss distributions.
    """
    if "channel_id" not in comparison_df.columns:
        return
    
    channel_data = comparison_df.filter(pl.col("channel_id").is_not_null())
    if channel_data.height == 0:
        return
    
    # Distribution of differences by channel
    plt.figure(figsize=(14, 8))
    splits = channel_data["split"].unique().to_list()
    n_splits = len(splits)
    
    for i, split in enumerate(splits):
        split_data = channel_data.filter(pl.col("split") == split)
        
        # Create subplot for each split
        plt.subplot(n_splits, 1, i + 1)
        
        # Violin plot by channel
        plot_df = split_data.select(["channel_id", "diff"]).to_pandas()
        sns.violinplot(data=plot_df, x="channel_id", y="diff")
        plt.title(f"Loss Difference Distribution by Channel ({split})")
        plt.xlabel("Channel")
        plt.ylabel("Loss Difference")
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label="No Change")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "channel_distribution.png", dpi=150)
    plt.close()

def plot_channel_ranking(comparison_df: pl.DataFrame, output_dir: Path):
    """Plot channels ranked by improvement."""
    channel_stats = compute_channel_stats(comparison_df)
    if channel_stats.height == 0:
        return
    
    # Get target split
    splits = channel_stats["split"].unique().to_list()
    target_split = "test" if "test" in splits else splits[0]
    
    split_stats = channel_stats.filter(pl.col("split") == target_split)
    
    # Sort by mean improvement (most negative = best)
    split_stats = split_stats.sort("mean_diff")
    
    plt.figure(figsize=(12, max(8, len(split_stats) * 0.3)))
    plot_data = split_stats.to_pandas()
    sns.barplot(data=plot_data, x="mean_diff", y="channel_id", palette="RdYlGn_r")
    plt.title(f"Channel Ranking by Mean Improvement ({target_split} split)")
    plt.xlabel("Mean Loss Difference (Negative = Better)")
    plt.ylabel("Channel")
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "channel_ranking.png", dpi=150)
    plt.close()

def plot_channel_improvement_heatmap(comparison_df: pl.DataFrame, output_dir: Path):
    """Create heatmap showing improvement by channel and entity."""
    if "channel_id" not in comparison_df.columns or "entity_id" not in comparison_df.columns:
        return
    
    channel_data = comparison_df.filter(
        pl.col("channel_id").is_not_null() & 
        pl.col("entity_id").is_not_null()
    )
    if channel_data.height == 0:
        return
    
    # Compute mean improvement by channel and entity
    heatmap_data = channel_data.group_by(["split", "channel_id", "entity_id"]).agg([
        pl.col("diff").mean().alias("mean_diff")
    ])
    
    # Get target split
    splits = heatmap_data["split"].unique().to_list()
    target_split = "test" if "test" in splits else splits[0]
    
    split_data = heatmap_data.filter(pl.col("split") == target_split)
    
    # Pivot for heatmap
    pivot_data = split_data.pivot(
        index="channel_id",
        columns="entity_id",
        values="mean_diff"
    ).to_pandas().set_index("channel_id")
    
    plt.figure(figsize=(max(12, pivot_data.shape[1] * 0.5), max(8, pivot_data.shape[0] * 0.4)))
    sns.heatmap(pivot_data, annot=True, fmt=".4f", cmap="RdYlGn_r", center=0, 
                cbar_kws={"label": "Mean Loss Difference"})
    plt.title(f"Channel × Entity Improvement Heatmap ({target_split} split)")
    plt.xlabel("Entity")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.savefig(output_dir / "channel_entity_heatmap.png", dpi=150)
    plt.close()

def plot_channel_correlation(comparison_df: pl.DataFrame, output_dir: Path):
    """Create correlation matrix of improvements across channels."""
    if "channel_id" not in comparison_df.columns or "sample_id" not in comparison_df.columns:
        return
    
    channel_data = comparison_df.filter(pl.col("channel_id").is_not_null())
    if channel_data.height == 0:
        return
    
    # Get target split
    splits = channel_data["split"].unique().to_list()
    target_split = "test" if "test" in splits else splits[0]
    
    split_data = channel_data.filter(pl.col("split") == target_split)
    
    # Pivot: sample_id × channel_id with diff values
    pivot_data = split_data.pivot(
        index="sample_id",
        columns="channel_id",
        values="diff"
    )
    
    # Compute correlation matrix
    correlation = pivot_data.select([col for col in pivot_data.columns if col != "sample_id"]).to_pandas().corr()
    
    plt.figure(figsize=(max(10, correlation.shape[0] * 0.5), max(8, correlation.shape[1] * 0.5)))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"label": "Correlation"})
    plt.title(f"Channel Improvement Correlation Matrix ({target_split} split)")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.savefig(output_dir / "channel_correlation.png", dpi=150)
    plt.close()

def plot_entity_improvement_map(comparison_df: pl.DataFrame, output_dir: Path):
    """Plot entity improvement map (existing functionality, improved)."""
    # Aggregate to sample level
    sample_df = aggregate_to_sample_level(comparison_df)
    
    if "entity_id" not in sample_df.columns:
        return
    
    # Per-Entity Statistics
    entity_stats = sample_df.group_by(["split", "entity_id"]).agg([
        pl.col("diff").mean().alias("mean_diff"),
        pl.col("diff").count().alias("count")
    ]).sort("mean_diff")
    
    # Filter for test split if available, else first available
    splits = entity_stats["split"].unique().to_list()
    target_split = "test" if "test" in splits else splits[0]
    
    split_stats = entity_stats.filter(pl.col("split") == target_split)
    
    # Top and bottom entities
    top_entities = split_stats.head(20)
    bottom_entities = split_stats.tail(20)
    
    plt.figure(figsize=(12, 8))
    plot_data = pl.concat([top_entities, bottom_entities]).to_pandas()
    sns.barplot(data=plot_data, x="mean_diff", y="entity_id")
    plt.title(f"Mean Difference by Entity ({target_split} split) - Top/Bottom 20")
    plt.xlabel("Mean Loss Difference")
    plt.tight_layout()
    plt.savefig(output_dir / "entity_improvement_map.png", dpi=150)
    plt.close()

# ============================================================================
# Main Command
# ============================================================================

def _load_and_prepare_data(
    baseline: str,
    candidate: str,
    split: str,
    epoch: Optional[int],
    channels: Optional[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and prepare baseline and candidate dataframes."""
    print(f"Loading baseline from {baseline}...")
    baseline_df = load_metrics(baseline, split=split, epoch=epoch)
    
    print(f"Loading candidate from {candidate}...")
    candidate_df = load_metrics(candidate, split=split, epoch=epoch)
    
    # Filter channels if requested
    if channels:
        channel_list = [c.strip() for c in channels.split(",")]
        print(f"Filtering to channels: {channel_list}")
        baseline_df = filter_channels(baseline_df, channel_list)
        candidate_df = filter_channels(candidate_df, channel_list)
    
    return baseline_df, candidate_df

def _determine_analysis_levels(
    baseline_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    level: str
) -> tuple[Optional[str], Optional[str]]:
    """Determine which analysis levels to perform."""
    if level == "both":
        return "sample", "channel"
    elif level == "auto":
        # Detect from data
        baseline_level = detect_data_level(baseline_df)
        candidate_level = detect_data_level(candidate_df)
        if baseline_level == "channel" and candidate_level == "channel":
            return "sample", "channel"
        else:
            return "sample", None
    else:
        sample_level = level if level == "sample" else None
        channel_level = level if level == "channel" else None
        return sample_level, channel_level

def _run_sample_level_analysis(
    baseline_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    output_dir: Path
) -> None:
    """Run sample-level comparison analysis."""
    print("  Computing sample-level comparison...")
    sample_comparison = compute_comparison(baseline_df, candidate_df, level="sample")
    
    # Sample-level statistics
    stats = sample_comparison.group_by("split").agg([
        pl.col("diff").mean().alias("mean_diff"),
        pl.col("diff").median().alias("median_diff"),
        pl.col("diff").std().alias("std_diff"),
        (pl.col("diff") < 0).mean().alias("improvement_rate"),
        pl.col("diff").count().alias("count")
    ])
    
    print("\nGlobal Comparison Statistics (Sample-level):")
    print(stats)
    stats.write_csv(output_dir / "global_stats.csv")
    
    # Per-Entity Statistics
    if "entity_id" in sample_comparison.columns:
        entity_stats = sample_comparison.group_by(["split", "entity_id"]).agg([
            pl.col("diff").mean().alias("mean_diff"),
            pl.col("diff").count().alias("count")
        ]).sort("mean_diff")
        entity_stats.write_csv(output_dir / "entity_stats.csv")
    
    # Sample-level visualizations
    plot_sample_distributions(sample_comparison, output_dir)
    plot_entity_improvement_map(sample_comparison, output_dir)

def _run_channel_level_analysis(
    baseline_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    output_dir: Path
) -> None:
    """Run channel-level comparison analysis."""
    print("  Computing channel-level comparison...")
    channel_comparison = compute_comparison(baseline_df, candidate_df, level="channel")
    
    # Channel-level statistics
    channel_stats = compute_channel_stats(channel_comparison)
    if channel_stats.height > 0:
        print("\nChannel Comparison Statistics:")
        print(channel_stats)
        channel_stats.write_csv(output_dir / "channel_stats.csv")
        
        # Channel × Entity statistics
        if "entity_id" in channel_comparison.columns:
            channel_entity_stats = channel_comparison.filter(
                pl.col("channel_id").is_not_null() & 
                pl.col("entity_id").is_not_null()
            ).group_by(["split", "channel_id", "entity_id"]).agg([
                pl.col("diff").mean().alias("mean_diff"),
                pl.col("diff").count().alias("count")
            ]).sort("mean_diff")
            channel_entity_stats.write_csv(output_dir / "channel_entity_stats.csv")
        
        # Channel-level visualizations
        plot_channel_distributions(channel_comparison, output_dir)
        plot_channel_ranking(channel_comparison, output_dir)
        plot_channel_improvement_heatmap(channel_comparison, output_dir)
        plot_channel_correlation(channel_comparison, output_dir)
    else:
        print("  No channel-level data found, skipping channel analysis")

@app.command()
def compare(
    baseline: str = typer.Option(..., help="Path to baseline experiment output directory"),
    candidate: str = typer.Option(..., help="Path to candidate experiment output directory"),
    output: str = typer.Option(..., help="Directory to save analysis results"),
    split: Optional[str] = typer.Option(None, help="Split to compare: 'train', 'val', 'test' (default: 'test')"),
    epoch: Optional[int] = typer.Option(None, help="Epoch to load (applies to all splits in new format, default: latest)"),
    level: str = typer.Option("auto", help="Analysis level: 'sample', 'channel', 'both', or 'auto'"),
    channels: Optional[str] = typer.Option(None, help="Comma-separated channel IDs to filter"),
):
    """
    Compare per-sample performance between two experiments.
    Supports both sample-level and per-channel analysis.
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default to test split if not specified
    if split is None:
        split = "test"
    
    # Load and prepare data
    baseline_df, candidate_df = _load_and_prepare_data(
        baseline, candidate, split, epoch, channels
    )
    
    # Determine analysis levels
    sample_level, channel_level = _determine_analysis_levels(
        baseline_df, candidate_df, level
    )
    
    print("Computing comparison...")
    
    # Run sample-level analysis
    if sample_level:
        _run_sample_level_analysis(baseline_df, candidate_df, output_dir)
    
    # Run channel-level analysis
    if channel_level:
        _run_channel_level_analysis(baseline_df, candidate_df, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    app()
