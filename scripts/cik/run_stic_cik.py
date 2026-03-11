from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cik_adapter import DEFAULT_CONTEXT_FIELDS, build_cik_dataloader  # noqa: E402
from utils.cik_stic import (  # noqa: E402
    STICCiKPrototype,
    compute_basic_metrics,
    describe_positions,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the thin CiK STIC runner."""

    parser = argparse.ArgumentParser(
        description="Run a thin STIC-style prototype on the CiK Hugging Face dataset."
    )
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--context-fields",
        nargs="+",
        default=list(DEFAULT_CONTEXT_FIELDS),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gate-floor", type=float, default=0.05)
    parser.add_argument("--gate-context-scale", type=float, default=1.0)
    parser.add_argument("--gate-delta-scale", type=float, default=1.5)
    parser.add_argument("--gate-bias", type=float, default=-2.4)
    parser.add_argument("--noise-scale-factor", type=float, default=0.15)
    parser.add_argument("--show-rows", type=int, default=5)
    parser.add_argument("--show-text-chars", type=int, default=220)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    return parser.parse_args()


def truncate_text(text: str, max_chars: int) -> str:
    """Return a single-line preview of the merged context text."""

    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)] + "..."


def summarize_constraint_text(context_fields: Dict[str, str], max_chars: int) -> str:
    """Return a short constraint summary if the current task exposes one."""

    text = context_fields.get("constraints", "").strip()
    if not text:
        return "none"
    return truncate_text(text, max_chars=max_chars)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested device, falling back to CPU when CUDA is unavailable."""

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def summarize_batch_prediction_shape(outputs: List[Any]) -> Any:
    """Return a compact batch shape summary even when horizons differ across samples."""

    if not outputs:
        return ()
    horizons = {int(output.pred.shape[0]) for output in outputs}
    if len(horizons) == 1:
        horizon = next(iter(horizons))
        return (len(outputs), horizon, 1)
    return [tuple(output.pred.reshape(-1, 1).shape) for output in outputs]


def build_record(
    sample: Dict[str, Any],
    output: Any,
    truth: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build one flat evaluation record for a CiK sample."""

    metrics = compute_basic_metrics(
        truth=truth,
        pred=output.pred,
        pred_h=output.pred_h,
        pred_c=output.pred_c,
        gate=output.gate,
        region_of_interest=sample["region_of_interest"],
    )
    sample_paths = output.sample_paths(
        n_samples=args.max_samples,
        seed=args.seed + int(sample["row_index"]),
    )
    return {
        "row_index": sample["row_index"],
        "task_name": sample["task_name"],
        "rule_name": output.rule_name,
        "confidence": output.confidence,
        "seasonal_period": output.seasonal_period,
        "noise_scale": output.noise_scale,
        "context_sources": ",".join(sample["context_sources"]),
        "roi_size": len(sample["region_of_interest"]),
        "roi_positions": describe_positions(
            sample["region_of_interest"],
            horizon=int(truth.shape[0]),
        ),
        "applied_positions": describe_positions(
            output.applied_positions,
            horizon=int(truth.shape[0]),
        ),
        "text_preview": truncate_text(sample["context_text"], max_chars=args.show_text_chars),
        "constraint_summary": summarize_constraint_text(
            sample["context_fields"],
            max_chars=args.show_text_chars,
        ),
        "pred_h_shape": str(tuple(output.pred_h.reshape(-1, 1).shape)),
        "pred_c_shape": str(tuple(output.pred_c.reshape(-1, 1).shape)),
        "gate_shape": str(tuple(output.gate.reshape(-1, 1).shape)),
        "pred_shape": str(tuple(output.pred.reshape(-1, 1).shape)),
        "sample_shape": "x".join(str(dim) for dim in sample_paths.shape),
        "notes": " | ".join(output.notes),
        **metrics,
    }


def main() -> None:
    """Load CiK tasks, run the thin STIC prototype, and print debug/metric summaries."""

    args = parse_args()
    selected_preset = (
        args.preset
        if args.preset is not None
        else (None if (args.task_name or args.task_names) else "stic_minimal")
    )
    device = resolve_device(args.device)

    dataset, dataloader = build_cik_dataloader(
        split=args.split,
        task_name=args.task_name,
        task_names=args.task_names,
        preset_name=selected_preset,
        limit=args.limit,
        cache_dir=args.cache_dir,
        context_fields=args.context_fields,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
    )
    if len(dataset) == 0:
        raise SystemExit("No CiK rows matched the requested selection.")

    model = STICCiKPrototype(
        gate_floor=args.gate_floor,
        gate_context_scale=args.gate_context_scale,
        gate_delta_scale=args.gate_delta_scale,
        gate_bias=args.gate_bias,
        noise_scale_factor=args.noise_scale_factor,
    )

    first_batch_debug: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []
    for batch in dataloader:
        history_batch = batch["history"].to(device)
        future_batch = batch["future"].to(device)
        history_mask = batch["history_mask"].to(device)
        future_mask = batch["future_mask"].to(device)

        batch_records: List[Dict[str, Any]] = []
        batch_outputs = []
        batch_size = len(batch["task_name"])
        for item_index in range(batch_size):
            future_length = int(batch["future_lengths"][item_index].item())
            history_length = int(batch["history_lengths"][item_index].item())
            sample = {
                "row_index": batch["row_index"][item_index],
                "task_name": batch["task_name"][item_index],
                "history_frame": batch["history_frame"][item_index],
                "future_frame": batch["future_frame"][item_index],
                "context_text": batch["context_text"][item_index],
                "context_fields": batch["context_fields"][item_index],
                "context_sources": batch["context_sources"][item_index],
                "region_of_interest": batch["region_of_interest"][item_index],
                "history_length": history_length,
                "future_length": future_length,
            }
            output = model.forward(
                history_frame=sample["history_frame"],
                future_frame=sample["future_frame"],
                context_text=sample["context_text"],
                task_name=sample["task_name"],
            )
            truth = future_batch[item_index, :future_length, 0].detach().cpu().numpy()
            batch_records.append(build_record(sample=sample, output=output, truth=truth, args=args))
            batch_outputs.append(output)

        if first_batch_debug is None and batch_outputs:
            first_batch_debug = {
                "batch_history_shape": tuple(history_batch.shape),
                "batch_future_shape": tuple(future_batch.shape),
                "batch_history_mask_shape": tuple(history_mask.shape),
                "batch_future_mask_shape": tuple(future_mask.shape),
                "batch_history_lengths": batch["history_lengths"].tolist(),
                "batch_future_lengths": batch["future_lengths"].tolist(),
                "pred_h_shape": summarize_batch_prediction_shape(batch_outputs),
                "pred_c_shape": summarize_batch_prediction_shape(batch_outputs),
                "gate_shape": summarize_batch_prediction_shape(batch_outputs),
                "pred_shape": summarize_batch_prediction_shape(batch_outputs),
                "sample_text_preview": batch_records[0]["text_preview"],
                "sample_effect_summary": batch_records[0]["notes"] or "none",
                "sample_roi_summary": batch_records[0]["roi_positions"],
                "sample_constraint_summary": batch_records[0]["constraint_summary"],
            }

        records.extend(batch_records)

    results = pd.DataFrame.from_records(records)
    summary = (
        results.groupby("task_name", dropna=False)
        .agg(
            rows=("task_name", "size"),
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            mse_h=("mse_h", "mean"),
            mae_h=("mae_h", "mean"),
            mse_c=("mse_c", "mean"),
            mae_c=("mae_c", "mean"),
            mse_gain_vs_h=("mse_gain_vs_h", "mean"),
            mae_gain_vs_h=("mae_gain_vs_h", "mean"),
            roi_mse=("roi_mse", "mean"),
            roi_mae=("roi_mae", "mean"),
            gate_mean=("gate_mean", "mean"),
            pred_c_delta_mean_abs=("pred_c_delta_mean_abs", "mean"),
            pred_final_delta_mean_abs=("pred_final_delta_mean_abs", "mean"),
        )
        .sort_values(["mae", "task_name"])
        .reset_index()
    )

    print(f"device={device}")
    print(f"rows={len(results)}")
    print(f"unique_tasks={results['task_name'].nunique()}")
    print(f"preset={selected_preset}")

    if first_batch_debug is not None:
        print()
        print("Batch debug")
        for key, value in first_batch_debug.items():
            print(f"{key}={value}")

    print()
    print("Per-task summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print()
    print("Sample rows")
    display_columns = [
        "row_index",
        "task_name",
        "rule_name",
        "confidence",
        "mse_h",
        "mse_c",
        "mse",
        "mae_h",
        "mae_c",
        "mae",
        "gate_mean",
        "applied_positions",
        "roi_positions",
        "text_preview",
        "notes",
    ]
    print(
        results.loc[:, display_columns]
        .head(max(0, args.show_rows))
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print()
        print(f"saved_rows={output_path}")

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
