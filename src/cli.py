import argparse
import os
import sys

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from tqdm import tqdm

from src.config_parser import load_config
from src.dataset_loader import load_dataset
from src.model_runner import HuggingFaceRunner
from src.profiler import profile_inference
from src.reporter import generate_visualizations, print_summary_table, save_results_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM Performance Benchmarking CLI Tool (Hugging Face)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and dataset path without running inference.",
    )
    return parser


def run_benchmark(config, prompts) -> None:
    records = []

    print("\n" + "=" * 65)
    print("           STARTING LLM BENCHMARK SUITE")
    print("=" * 65)
    print(f" Models to evaluate: {len(config.models)}")
    print(f" Prompts per model:  {len(prompts)}")
    print(f" Max tokens / run:   {config.max_new_tokens}")
    print(f" Target device:      {config.device}")
    print("=" * 65 + "\n")

    for model_idx, model_id in enumerate(config.models, 1):
        print(f"[{model_idx}/{len(config.models)}] Loading model: {model_id} ...")

        runner = None
        try:
            runner = HuggingFaceRunner(model_id=model_id, device=config.device)
        except Exception as exc:
            print(f" [ERROR] Failed to load model '{model_id}': {exc}. Skipping to next.")
            continue

        # Optional single-token warm-up to initialize internal caches/kernels
        if config.warmup:
            try:
                runner.generate("Warmup prompt", max_tokens=2, temperature=0.0)
            except Exception:
                pass

        # Process prompts with progress bar
        pbar = tqdm(prompts, desc=f" Benchmarking {model_id.split('/')[-1]}", unit="prompt", ncols=80)
        for item in pbar:
            prompt_id = item["id"]
            prompt_text = item["prompt"]

            try:
                metrics = profile_inference(
                    lambda: runner.generate(
                        prompt_text,
                        max_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                    ),
                    min_quality_tokens=config.min_quality_tokens,
                )
                metrics["model_name"] = model_id
                metrics["prompt_id"] = prompt_id
                metrics["prompt_text"] = prompt_text
                records.append(metrics)
            except Exception as exc:
                print(f"\n [ERROR] Inference failed for {model_id} on prompt '{prompt_id}': {exc}")

        # Free memory before loading the next model
        runner.cleanup()
        print(f" Finished {model_id} and freed memory.\n")

    if not records:
        print("[WARN] No benchmark records were collected. Check model loading logs.")
        return

    df = pd.DataFrame(records)

    # Persist CSV and generate charts
    csv_file = save_results_csv(records, config.output_dir)
    chart_files = generate_visualizations(df, config.output_dir)

    # Print summary table to console
    print_summary_table(df)

    print("Benchmarking artifacts saved successfully:")
    print(f" -> CSV Report:     {csv_file}")
    for chart in chart_files:
        print(f" -> Visualization:  {chart}")
    print()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        prompts = load_dataset(config.dataset_path)
    except Exception as exc:
        print(f"[ERROR] Configuration or dataset error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" LLM Benchmarking Suite - Configuration Verified")
    print("=" * 60)
    print(f" Models ({len(config.models)}):       {', '.join(config.models)}")
    print(f" Dataset Path:     {config.dataset_path} ({len(prompts)} prompts loaded)")
    print(f" Output Directory: {config.output_dir}")
    print(f" Max New Tokens:   {config.max_new_tokens}")
    print(f" Temperature:      {config.temperature}")
    print(f" Target Device:    {config.device}")
    print(f" Warmup Enabled:   {config.warmup}")
    print("=" * 60)

    if args.dry_run:
        print("Dry run completed successfully. No models were loaded.")
        return

    run_benchmark(config, prompts)


if __name__ == "__main__":
    main()
