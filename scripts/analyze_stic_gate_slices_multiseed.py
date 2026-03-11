from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import analyze_stic_gate_slices as single_seed


@dataclass(frozen=True)
class ModelGroup:
    """Seed-aligned checkpoint group for a single gate-input family."""

    label: str
    specs: Tuple[single_seed.ModelSpec, ...]


def parse_args() -> argparse.Namespace:
    """Parse multi-seed slice-analysis arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate utility, horizon, and corruption slice metrics across multiple "
            "seed-aligned STIC checkpoints."
        )
    )
    parser.add_argument(
        "--model_group",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Format: label=model_id1:gate_mode,model_id2:gate_mode,... "
            "All groups must have the same number of specs."
        ),
    )
    parser.add_argument(
        "--reference_label",
        type=str,
        default="g1b",
        help="Model group used to define the clean utility terciles per seed.",
    )
    parser.add_argument(
        "--seed_set",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023],
        help="Deterministic corruption seeds aligned with the model-group order.",
    )
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--features", type=str, default="MS")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--stic_target_index", type=int, default=-1)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--stic_gate_hidden_feat_dim", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def parse_model_group(spec: str) -> ModelGroup:
    """Parse a `label=model_id:gate_mode,...` model-group specification."""
    label, remainder = spec.split("=", 1)
    specs = tuple(
        single_seed.parse_model_spec(f"{label_seed}={entry}")
        for label_seed, entry in [
            (f"{label}@{index}", item.strip())
            for index, item in enumerate(remainder.split(","))
        ]
    )
    return ModelGroup(label=label, specs=specs)


def validate_groups(model_groups: Sequence[ModelGroup], seed_set: Sequence[int]) -> None:
    """Validate that all model groups share the same seed-aligned cardinality."""
    if not model_groups:
        raise ValueError("At least one model group is required.")
    expected_size = len(model_groups[0].specs)
    if expected_size != len(seed_set):
        raise ValueError(
            f"Expected {expected_size} seeds to match the first model group, got {len(seed_set)}."
        )
    for model_group in model_groups:
        if len(model_group.specs) != expected_size:
            raise ValueError(
                "All model groups must contain the same number of seed-aligned specs."
            )


def aggregate_nested_metrics(
    summaries: Dict[str, List[Dict[str, Dict[str, float]]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate nested metric dictionaries into mean/std summaries."""
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, per_seed_tables in summaries.items():
        aggregated[label] = {}
        group_names = per_seed_tables[0].keys()
        for group_name in group_names:
            aggregated[label][group_name] = {}
            metric_names = per_seed_tables[0][group_name].keys()
            for metric_name in metric_names:
                values = np.asarray(
                    [table[group_name][metric_name] for table in per_seed_tables],
                    dtype=np.float64,
                )
                aggregated[label][group_name][f"{metric_name}_mean"] = float(values.mean())
                aggregated[label][group_name][f"{metric_name}_std"] = float(values.std())
    return aggregated


def format_mean_std(value_mean: float, value_std: float) -> str:
    """Format a metric pair as `mean±std` with four decimals."""
    return f"{value_mean:.4f}±{value_std:.4f}"


def print_aggregate_summary(
    title: str, summary: Dict[str, Dict[str, Dict[str, float]]]
) -> None:
    """Print a compact aggregate summary table."""
    print(f"\n[{title}]")
    for label, metrics_by_group in summary.items():
        print(label)
        for group_name, metrics in metrics_by_group.items():
            ordered_metric_names = sorted(
                {metric_name.rsplit("_", 1)[0] for metric_name in metrics.keys()}
            )
            parts: List[str] = []
            for metric_name in ordered_metric_names:
                parts.append(
                    f"{metric_name}="
                    f"{format_mean_std(metrics[f'{metric_name}_mean'], metrics[f'{metric_name}_std'])}"
                )
            print(f"  {group_name}: {', '.join(parts)}")


def build_seed_cli_args(base_args: argparse.Namespace, seed: int) -> argparse.Namespace:
    """Clone CLI args with the per-seed corruption seed applied."""
    seed_args = argparse.Namespace(**vars(base_args))
    seed_args.seed = int(seed)
    return seed_args


def load_checkpoint_to_experiment(
    exp: single_seed.Exp_Long_Term_Forecast, checkpoint_path: str
) -> None:
    """Load a checkpoint through CPU first to avoid CUDA index mismatches."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    exp.model.load_state_dict(state_dict)
    exp.model.to(exp.device)
    exp.model.eval()


def main() -> None:
    """Run multi-seed slice analysis for seed-aligned STIC checkpoint groups."""
    cli_args = parse_args()
    model_groups = [parse_model_group(spec) for spec in cli_args.model_group]
    validate_groups(model_groups, cli_args.seed_set)

    if cli_args.reference_label not in {model_group.label for model_group in model_groups}:
        raise ValueError(
            f"reference_label={cli_args.reference_label} not found in model groups."
        )

    utility_tables: Dict[str, List[Dict[str, Dict[str, float]]]] = {
        model_group.label: [] for model_group in model_groups
    }
    horizon_tables: Dict[str, List[Dict[str, Dict[str, float]]]] = {
        model_group.label: [] for model_group in model_groups
    }
    corruption_tables: Dict[str, List[Dict[str, Dict[str, float]]]] = {
        model_group.label: [] for model_group in model_groups
    }
    checkpoint_log: Dict[str, List[str]] = {model_group.label: [] for model_group in model_groups}

    for seed_index, seed in enumerate(cli_args.seed_set):
        single_seed.set_seed(int(seed))
        seed_args = build_seed_cli_args(cli_args, int(seed))
        exp_map: Dict[str, single_seed.Exp_Long_Term_Forecast] = {}

        for model_group in model_groups:
            model_spec = model_group.specs[seed_index]
            exp_args = single_seed.build_exp_args(seed_args, model_spec)
            exp = single_seed.Exp_Long_Term_Forecast(exp_args)
            checkpoint_setting = single_seed.resolve_checkpoint_setting(
                exp_args.checkpoints, model_spec.model_id
            )
            checkpoint_path = os.path.join(
                exp_args.checkpoints, checkpoint_setting, "checkpoint.pth"
            )
            load_checkpoint_to_experiment(exp, checkpoint_path)
            exp_map[model_group.label] = exp
            checkpoint_log[model_group.label].append(checkpoint_setting)

        reference_exp = exp_map[cli_args.reference_label]
        _, reference_loader = reference_exp._get_data("test")
        sample_slice_labels = single_seed.collect_reference_slice_labels(
            reference_exp, reference_loader, cli_args.max_batches
        )

        for model_group in model_groups:
            exp = exp_map[model_group.label]
            _, test_loader = exp._get_data("test")
            utility_summary, horizon_summary, corruption_summary = single_seed.analyze_model(
                exp=exp,
                data_loader=test_loader,
                sample_slice_labels=sample_slice_labels,
                max_batches=cli_args.max_batches,
                seed=int(seed),
                dropout_p=cli_args.dropout_p,
            )
            utility_tables[model_group.label].append(utility_summary)
            horizon_tables[model_group.label].append(horizon_summary)
            corruption_tables[model_group.label].append(corruption_summary)

    utility_aggregate = aggregate_nested_metrics(utility_tables)
    horizon_aggregate = aggregate_nested_metrics(horizon_tables)
    corruption_aggregate = aggregate_nested_metrics(corruption_tables)

    print("Resolved checkpoints by seed:")
    for label, checkpoint_settings in checkpoint_log.items():
        joined_settings = " | ".join(checkpoint_settings)
        print(f"  {label}: {joined_settings}")
    print_aggregate_summary("utility_slices_aggregate", utility_aggregate)
    print_aggregate_summary("horizon_slices_aggregate", horizon_aggregate)
    print_aggregate_summary("corruption_types_aggregate", corruption_aggregate)


if __name__ == "__main__":
    main()
