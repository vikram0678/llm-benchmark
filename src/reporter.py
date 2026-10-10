import os
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


def save_results_csv(records: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")

    df = pd.DataFrame(records)
    # Reorder columns cleanly if present
    preferred_order = [
        "model_name",
        "prompt_id",
        "latency_sec",
        "throughput_tps",
        "peak_ram_mb",
        "peak_vram_mb",
        "quality_ttr",
        "token_count",
        "is_degenerate",
        "prompt_text",
        "output_text",
    ]
    cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[cols]
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_visualizations(df: pd.DataFrame, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = []

    # Calculate grouped means
    summary = df.groupby("model_name").agg({
        "latency_sec": "mean",
        "throughput_tps": "mean",
        "peak_ram_mb": "max",
        "peak_vram_mb": "max",
    }).reset_index()

    models = summary["model_name"].tolist()

    # 1. Average Latency Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, summary["latency_sec"], color="#2b5c8f", width=0.5)
    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("Average Inference Latency by Model (Lower is Better)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.2fs", padding=3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    latency_plot_path = os.path.join(output_dir, "latency_comparison.png")
    plt.savefig(latency_plot_path, dpi=200)
    plt.close()
    generated_plots.append(latency_plot_path)

    # 2. Peak Memory Usage Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    has_vram = summary["peak_vram_mb"].max() > 0

    if has_vram:
        x = range(len(models))
        width = 0.35
        ax.bar([p - width / 2 for p in x], summary["peak_ram_mb"], width, label="RAM (RSS)", color="#3e8e7e")
        ax.bar([p + width / 2 for p in x], summary["peak_vram_mb"], width, label="VRAM (CUDA)", color="#c7522a")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend()
    else:
        bars = ax.bar(models, summary["peak_ram_mb"], color="#3e8e7e", width=0.5)
        ax.bar_label(bars, fmt="%.1f MB", padding=3)
        plt.xticks(rotation=15, ha="right")

    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage by Model (Lower is Better)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    memory_plot_path = os.path.join(output_dir, "memory_comparison.png")
    plt.savefig(memory_plot_path, dpi=200)
    plt.close()
    generated_plots.append(memory_plot_path)

    # 3. Throughput Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, summary["throughput_tps"], color="#5c3c92", width=0.5)
    ax.set_ylabel("Throughput (Tokens / Second)")
    ax.set_title("Generation Throughput by Model (Higher is Better)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.1f TPS", padding=3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    throughput_plot_path = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(throughput_plot_path, dpi=200)
    plt.close()
    generated_plots.append(throughput_plot_path)

    return generated_plots


def print_summary_table(df: pd.DataFrame) -> None:
    agg_df = df.groupby("model_name").agg(
        avg_latency=("latency_sec", "mean"),
        avg_throughput=("throughput_tps", "mean"),
        peak_ram_mb=("peak_ram_mb", "max"),
        peak_vram_mb=("peak_vram_mb", "max"),
        avg_quality_ttr=("quality_ttr", "mean"),
        degenerate_rate=("is_degenerate", lambda s: f"{(s.sum() / len(s)) * 100:.1f}%"),
    ).reset_index()

    agg_df["avg_latency"] = agg_df["avg_latency"].apply(lambda v: f"{v:.3f} s")
    agg_df["avg_throughput"] = agg_df["avg_throughput"].apply(lambda v: f"{v:.2f} TPS")
    agg_df["peak_ram_mb"] = agg_df["peak_ram_mb"].apply(lambda v: f"{v:.1f} MB")
    agg_df["peak_vram_mb"] = agg_df["peak_vram_mb"].apply(lambda v: f"{v:.1f} MB")
    agg_df["avg_quality_ttr"] = agg_df["avg_quality_ttr"].apply(lambda v: f"{v:.3f}")

    agg_df.rename(columns={
        "model_name": "Model",
        "avg_latency": "Avg Latency",
        "avg_throughput": "Throughput",
        "peak_ram_mb": "Peak RAM",
        "peak_vram_mb": "Peak VRAM",
        "avg_quality_ttr": "Lexical TTR",
        "degenerate_rate": "Degenerate %",
    }, inplace=True)

    table_str = tabulate(agg_df, headers="keys", tablefmt="github", showindex=False)
    table_width = max(len(line) for line in table_str.splitlines())

    print("\n" + "=" * table_width)
    print("BENCHMARK SUMMARY RESULTS".center(table_width))
    print("=" * table_width)
    print(table_str)
    print("=" * table_width + "\n")
