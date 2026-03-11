from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cik_adapter import (  # noqa: E402
    DEFAULT_CONTEXT_FIELDS,
    build_cik_dataloader,
    build_cik_official_evaluator,
)
from utils.cik_stic import (  # noqa: E402
    STICCiKPrototype,
    compute_basic_metrics,
    describe_positions,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the thin CiK STIC runner."""

    parser = argparse.ArgumentParser(
        description="Run the thin STIC-style prototype on selected CiK tasks."
    )
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--limit-per-task",
        action="store_true",
        help="Apply --limit to each selected task independently.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="both",
        choices=("debug", "official", "both"),
    )
    parser.add_argument("--num-sample-paths", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Deprecated alias for --num-sample-paths.",
    )
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
    parser.add_argument("--summary-json", type=str, default=None)
    return parser.parse_args()


def truncate_text(text: str, max_chars: int) -> str:
    """Return a compact one-line preview of merged context text."""

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
        return (len(outputs), next(iter(horizons)), 1)
    return [tuple(output.pred.reshape(-1, 1).shape) for output in outputs]


def compute_official_branch_metrics(
    sample: Dict[str, Any],
    output: Any,
    num_sample_paths: int,
    seed: int,
) -> Dict[str, float]:
    """Evaluate history-only, context-aware, and gated branches with official CiK metrics."""

    evaluator = build_cik_official_evaluator(sample)
    branch_map = {
        "h": "pred_h",
        "c": "pred_c",
        "final": "pred_final",
    }
    metrics: Dict[str, float] = {
        "official_roi_weight": float(evaluator.roi_weight),
    }
    for suffix, branch_name in branch_map.items():
        branch_samples = output.sample_paths_for_branch(
            branch_name=branch_name,
            n_samples=num_sample_paths,
            seed=seed,
        )
        evaluation = evaluator.evaluate(branch_samples)
        for key, value in evaluation.items():
            metrics[f"official_{suffix}_{key}"] = float(value)
    metrics["official_metric_name"] = evaluator.metric_name
    metrics["official_gain_final_vs_h"] = (
        metrics["official_h_metric"] - metrics["official_final_metric"]
    )
    metrics["official_gain_final_vs_c"] = (
        metrics["official_c_metric"] - metrics["official_final_metric"]
    )
    return metrics


def build_record(
    sample: Dict[str, Any],
    output: Any,
    truth: np.ndarray,
    num_sample_paths: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build one flat evaluation record for a CiK sample."""

    metrics: Dict[str, Any] = {
        "gate_mean": float(np.mean(output.gate)),
        "gate_std": float(np.std(output.gate)),
        "pred_c_delta_mean_abs": float(np.mean(np.abs(output.pred_c - output.pred_h))),
        "pred_final_delta_mean_abs": float(np.mean(np.abs(output.pred - output.pred_h))),
    }
    if args.eval_mode in ("debug", "both"):
        metrics.update(
            compute_basic_metrics(
                truth=truth,
                pred=output.pred,
                pred_h=output.pred_h,
                pred_c=output.pred_c,
                gate=output.gate,
                region_of_interest=sample["region_of_interest"],
            )
        )
    if args.eval_mode in ("official", "both"):
        metrics.update(
            compute_official_branch_metrics(
                sample=sample,
                output=output,
                num_sample_paths=num_sample_paths,
                seed=args.seed + int(sample["row_index"]),
            )
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
        "sample_shape": f"{num_sample_paths}x{output.pred.shape[0]}x1",
        "notes": " | ".join(output.notes),
        **metrics,
    }


def build_summary(results: pd.DataFrame, eval_mode: str) -> Dict[str, pd.DataFrame]:
    """Build grouped summaries for debug and official evaluation outputs."""

    summaries: Dict[str, pd.DataFrame] = {}
    if eval_mode in ("debug", "both"):
        debug_summary = (
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
                gate_roi_mean=("gate_roi_mean", "mean"),
                gate_non_roi_mean=("gate_non_roi_mean", "mean"),
                pred_c_delta_mean_abs=("pred_c_delta_mean_abs", "mean"),
                pred_final_delta_mean_abs=("pred_final_delta_mean_abs", "mean"),
            )
            .sort_values(["mae", "task_name"])
            .reset_index()
        )
        summaries["debug"] = debug_summary

    if eval_mode in ("official", "both"):
        official_summary = (
            results.groupby("task_name", dropna=False)
            .agg(
                rows=("task_name", "size"),
                official_metric_name=("official_metric_name", "first"),
                official_h_metric=("official_h_metric", "mean"),
                official_c_metric=("official_c_metric", "mean"),
                official_final_metric=("official_final_metric", "mean"),
                official_gain_final_vs_h=("official_gain_final_vs_h", "mean"),
                official_gain_final_vs_c=("official_gain_final_vs_c", "mean"),
                official_h_crps=("official_h_crps", "mean"),
                official_c_crps=("official_c_crps", "mean"),
                official_final_crps=("official_final_crps", "mean"),
                official_h_roi_crps=("official_h_roi_crps", "mean"),
                official_c_roi_crps=("official_c_roi_crps", "mean"),
                official_final_roi_crps=("official_final_roi_crps", "mean"),
                official_h_violation_mean=("official_h_violation_mean", "mean"),
                official_c_violation_mean=("official_c_violation_mean", "mean"),
                official_final_violation_mean=("official_final_violation_mean", "mean"),
                gate_mean=("gate_mean", "mean") if "gate_mean" in results else ("confidence", "mean"),
            )
            .sort_values(["official_final_metric", "task_name"])
            .reset_index()
        )
        summaries["official"] = official_summary

    return summaries


def print_summaries(summaries: Dict[str, pd.DataFrame], eval_mode: str) -> None:
    """Print summary tables according to the selected evaluation mode."""

    if eval_mode in ("official", "both") and "official" in summaries:
        print()
        print("Official summary")
        print(
            summaries["official"].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    if eval_mode in ("debug", "both") and "debug" in summaries:
        print()
        print("Debug summary")
        print(
            summaries["debug"].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )


def print_sample_rows(results: pd.DataFrame, eval_mode: str, show_rows: int) -> None:
    """Print a compact sample-level preview table."""

    if show_rows <= 0:
        return

    base_columns = [
        "row_index",
        "task_name",
        "rule_name",
        "confidence",
    ]
    if eval_mode in ("official", "both"):
        base_columns.extend(
            [
                "official_h_metric",
                "official_c_metric",
                "official_final_metric",
                "official_gain_final_vs_c",
            ]
        )
    if eval_mode in ("debug", "both"):
        base_columns.extend(
            [
                "mse_h",
                "mse_c",
                "mse",
                "mae_h",
                "mae_c",
                "mae",
                "gate_mean",
            ]
        )
    base_columns.extend(
        [
            "applied_positions",
            "roi_positions",
            "text_preview",
            "notes",
        ]
    )

    display_columns = [column for column in base_columns if column in results.columns]
    print()
    print("Sample rows")
    print(
        results.loc[:, display_columns]
        .head(show_rows)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def save_summary_artifacts(
    results: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """Save row-level and grouped summaries to CSV/JSON if requested."""

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print()
        print(f"saved_rows={output_path}")

    if args.summary_csv:
        summary_frames = []
        for mode_name, frame in summaries.items():
            tagged_frame = frame.copy()
            tagged_frame.insert(0, "eval_mode", mode_name)
            summary_frames.append(tagged_frame)
        summary_frame = pd.concat(summary_frames, axis=0, ignore_index=True)
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(summary_path, index=False)
        print(f"saved_summary={summary_path}")

    if args.summary_json:
        json_path = Path(args.summary_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eval_mode": args.eval_mode,
            "num_sample_paths": resolve_num_sample_paths(args),
            "summaries": {
                mode_name: frame.to_dict(orient="records")
                for mode_name, frame in summaries.items()
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_summary_json={json_path}")


def resolve_num_sample_paths(args: argparse.Namespace) -> int:
    """Resolve the active number of forecast sample paths."""

    if args.max_samples is not None:
        return int(args.max_samples)
    return int(args.num_sample_paths)


def main() -> None:
    """Load CiK tasks, run the thin STIC prototype, and print eval summaries."""

    args = parse_args()
    num_sample_paths = resolve_num_sample_paths(args)
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
        limit_per_task=args.limit_per_task,
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
                "metric_scaling": float(batch["metric_scaling"][item_index].item()),
                "metadata": batch["metadata"][item_index],
            }
            output = model.forward(
                history_frame=sample["history_frame"],
                future_frame=sample["future_frame"],
                context_text=sample["context_text"],
                task_name=sample["task_name"],
            )
            truth = future_batch[item_index, :future_length, 0].detach().cpu().numpy()
            batch_records.append(
                build_record(
                    sample=sample,
                    output=output,
                    truth=truth,
                    num_sample_paths=num_sample_paths,
                    args=args,
                )
            )
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
    summaries = build_summary(results=results, eval_mode=args.eval_mode)

    print(f"device={device}")
    print(f"rows={len(results)}")
    print(f"unique_tasks={results['task_name'].nunique()}")
    print(f"preset={selected_preset}")
    print(f"limit_per_task={args.limit_per_task}")
    print(f"eval_mode={args.eval_mode}")
    print(f"num_sample_paths={num_sample_paths}")

    if first_batch_debug is not None:
        print()
        print("Batch debug")
        for key, value in first_batch_debug.items():
            print(f"{key}={value}")

    print_summaries(summaries=summaries, eval_mode=args.eval_mode)
    print_sample_rows(results=results, eval_mode=args.eval_mode, show_rows=args.show_rows)
    save_summary_artifacts(results=results, summaries=summaries, args=args)


if __name__ == "__main__":
    main()
