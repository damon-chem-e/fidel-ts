
import os
import typer
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List

app = typer.Typer()

def load_metrics(experiment_dir: str, epoch: Optional[int] = None) -> pl.DataFrame:
    """
    Load per-sample metrics from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment output directory.
        epoch: Specific epoch to load. If None, loads the last available epoch.
    """
    per_sample_dir = Path(experiment_dir) / "per_sample"
    if not per_sample_dir.exists():
        raise ValueError(f"No per-sample metrics found in {experiment_dir}")
        
    files = sorted(per_sample_dir.glob("epoch_*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {per_sample_dir}")
        
    if epoch is None:
        target_file = files[-1]
        print(f"Loading metrics from latest epoch: {target_file.name}")
    else:
        target_file = per_sample_dir / f"epoch_{epoch:03d}.parquet"
        if not target_file.exists():
            raise ValueError(f"Metrics for epoch {epoch} not found")
            
    return pl.read_parquet(target_file)

def compute_comparison(baseline_df: pl.DataFrame, candidate_df: pl.DataFrame):
    """
    Compare two dataframes by joining on entity_id and timestamp.
    """
    # Join on entity_id, timestamp, and split
    # We rename loss columns to distinguish them
    joined = baseline_df.join(
        candidate_df,
        on=["entity_id", "timestamp", "split"],
        suffix="_candidate"
    ).rename({"loss": "loss_baseline"})
    
    # Compute difference: candidate - baseline
    # Negative values mean candidate is better (lower loss)
    joined = joined.with_columns(
        (pl.col("loss_candidate") - pl.col("loss_baseline")).alias("diff")
    )
    
    return joined

@app.command()
def compare(
    baseline: str = typer.Option(..., help="Path to baseline experiment output directory"),
    candidate: str = typer.Option(..., help="Path to candidate experiment output directory"),
    output: str = typer.Option(..., help="Directory to save analysis results"),
    epoch: Optional[int] = typer.Option(None, help="Epoch number to compare (default: latest)"),
):
    """
    Compare per-sample performance between two experiments.
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading baseline from {baseline}...")
    baseline_df = load_metrics(baseline, epoch)
    
    print(f"Loading candidate from {candidate}...")
    candidate_df = load_metrics(candidate, epoch)
    
    print("Computing comparison...")
    comparison_df = compute_comparison(baseline_df, candidate_df)
    
    # 1. Global Statistics
    stats = comparison_df.group_by("split").agg([
        pl.col("diff").mean().alias("mean_diff"),
        pl.col("diff").median().alias("median_diff"),
        pl.col("diff").std().alias("std_diff"),
        (pl.col("diff") < 0).mean().alias("improvement_rate")
    ])
    
    print("\nGlobal Comparison Statistics:")
    print(stats)
    stats.write_csv(output_dir / "global_stats.csv")
    
    # 2. Per-Entity Statistics
    entity_stats = comparison_df.group_by(["split", "entity_id"]).agg([
        pl.col("diff").mean().alias("mean_diff"),
        pl.col("diff").count().alias("count")
    ]).sort("mean_diff")
    
    entity_stats.write_csv(output_dir / "entity_stats.csv")
    
    # 3. Visualizations
    sns.set_theme(style="whitegrid")
    
    # Distribution of differences
    plt.figure(figsize=(10, 6))
    for split in comparison_df["split"].unique():
        split_data = comparison_df.filter(pl.col("split") == split)
        sns.histplot(split_data["diff"], label=split, alpha=0.5, kde=True)
    
    plt.title("Distribution of Loss Differences (Candidate - Baseline)")
    plt.xlabel("Loss Difference (Negative = Candidate Better)")
    plt.legend()
    plt.savefig(output_dir / "diff_distribution.png")
    plt.close()
    
    # Improvement Map (Top 20 entities)
    # Filter for test split if available, else first available
    splits = comparison_df["split"].unique().to_list()
    target_split = "test" if "test" in splits else splits[0]
    
    top_entities = entity_stats.filter(pl.col("split") == target_split).head(20)
    bottom_entities = entity_stats.filter(pl.col("split") == target_split).tail(20)
    
    plt.figure(figsize=(12, 8))
    plot_data = pl.concat([top_entities, bottom_entities]).to_pandas()
    sns.barplot(data=plot_data, x="mean_diff", y="entity_id")
    plt.title(f"Mean Difference by Entity ({target_split} split) - Top/Bottom 20")
    plt.xlabel("Mean Loss Difference")
    plt.tight_layout()
    plt.savefig(output_dir / "entity_improvement_map.png")
    plt.close()
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    app()

