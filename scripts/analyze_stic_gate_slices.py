from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


@dataclass(frozen=True)
class ModelSpec:
    """Model identifier and gate mode needed to load a saved STIC checkpoint."""

    label: str
    model_id: str
    gate_input_mode: str


UTILITY_SLICE_NAMES: Tuple[str, str, str] = ("bottom", "middle", "top")
HORIZON_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("h1_24", 0, 24),
    ("h25_48", 24, 48),
    ("h49_72", 48, 72),
    ("h73_96", 72, 96),
)
CORRUPTION_MODES: Tuple[str, ...] = ("shuffle", "swap", "dropout")
CORRUPTION_MODE_OFFSET: Dict[str, int] = {
    "shuffle": 11,
    "swap": 23,
    "dropout": 37,
}


def parse_args() -> argparse.Namespace:
    """Parse analysis configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze utility slices, horizon slices, and corruption-specific tradeoffs "
            "for STIC gate inputs."
        )
    )
    parser.add_argument(
        "--model_spec",
        type=str,
        nargs="+",
        default=[
            "g0=stic_compact_g0_s2021:g0",
            "g1-lite=stic_compact_g1-lite_s2021:g1-lite",
            "g1b=stic_compact_g1b_s2021:g1b",
        ],
        help="Format: label=model_id:gate_mode",
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
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--stic_target_index", type=int, default=-1)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--stic_gate_hidden_feat_dim", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def parse_model_spec(spec: str) -> ModelSpec:
    """Parse a `label=model_id:gate_mode` specification."""
    label, remainder = spec.split("=", 1)
    parts = remainder.split(":")
    if len(parts) != 2:
        raise ValueError(
            "Model spec must follow label=model_id:gate_mode."
        )
    model_id, gate_input_mode = parts
    return ModelSpec(
        label=label,
        model_id=model_id,
        gate_input_mode=gate_input_mode,
    )


def infer_gate_hidden_scale(model_spec: ModelSpec) -> float:
    """Infer the hidden-summary scale for topclip-style gate modes from the model id."""
    if model_spec.gate_input_mode == "g1b-topclip":
        return 0.5
    if model_spec.gate_input_mode != "g1b-topclip-lite":
        return 1.0
    if "0p875" in model_spec.model_id:
        return 0.875
    if "0p75" in model_spec.model_id:
        return 0.75
    return 0.75


def infer_summary_reg_mode(model_spec: ModelSpec) -> str:
    """Infer the hidden-summary regularization mode from the gate-input mode."""
    if model_spec.gate_input_mode == "g1b-sumreg-rms":
        return "rms"
    if model_spec.gate_input_mode == "g1b-sumreg-clip":
        return "clip"
    return "none"


def set_seed(seed: int) -> None:
    """Set random seeds for deterministic evaluation-side corruption."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_exp_args(cli_args: argparse.Namespace, model_spec: ModelSpec) -> SimpleNamespace:
    """Build the minimal experiment args needed to load a saved checkpoint."""
    use_gpu = torch.cuda.is_available() and not cli_args.cpu
    gate_std_scale = 0.5 if model_spec.gate_input_mode == "g1b-meanheavy" else 1.0
    gate_hidden_scale = infer_gate_hidden_scale(model_spec)
    gate_summary_reg_mode = infer_summary_reg_mode(model_spec)
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=0,
        model_id=model_spec.model_id,
        model="STIC",
        data=cli_args.data,
        root_path=cli_args.root_path,
        data_path=cli_args.data_path,
        features=cli_args.features,
        target=cli_args.target,
        freq=cli_args.freq,
        checkpoints=cli_args.checkpoints,
        seq_len=cli_args.seq_len,
        label_len=cli_args.label_len,
        pred_len=cli_args.pred_len,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=cli_args.enc_in,
        dec_in=cli_args.dec_in,
        c_out=cli_args.c_out,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        num_workers=cli_args.num_workers,
        itr=1,
        train_epochs=1,
        batch_size=cli_args.batch_size,
        patience=3,
        learning_rate=1e-4,
        des="analysis",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=use_gpu,
        gpu=cli_args.gpu,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices=str(cli_args.gpu),
        device_ids=[cli_args.gpu],
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=cli_args.seed,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=16,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        individual=False,
        stic_mode="dynamic",
        stic_static_gate_value=0.5,
        stic_aux_weight=0.1,
        stic_gate_weight=0.1,
        stic_gate_target_mode="soft",
        stic_gate_soft_tau=0.02,
        stic_gate_input_mode=model_spec.gate_input_mode,
        stic_gate_hidden_feat_dim=cli_args.stic_gate_hidden_feat_dim,
        stic_gate_stats_mode="basic",
        stic_gate_std_scale=gate_std_scale,
        stic_gate_hidden_scale=gate_hidden_scale,
        stic_gate_summary_reg_mode=gate_summary_reg_mode,
        stic_gate_summary_clip_value=1.0,
        stic_target_index=cli_args.stic_target_index,
        stic_context_mixer_type="linear",
        stic_context_mixer_hidden_dim=0,
        stic_context_residual_scale=0.5,
        stic_context_corruption_mode="none",
        stic_context_corruption_prob=1.0,
        stic_context_dropout_p=cli_args.dropout_p,
        stic_context_corruption_gate_weight=1.0,
        stic_corrupt_context_aux_weight=0.0,
        stic_pair_rank_weight=0.0,
        stic_pair_rank_margin=0.05,
        alpha=0.1,
        top_p=0.5,
        pos=1,
    )


def resolve_checkpoint_setting(checkpoints_dir: str, model_id: str) -> str:
    """Find the single checkpoint directory that matches a model id."""
    checkpoint_root = Path(checkpoints_dir)
    matches = [
        candidate.name
        for candidate in checkpoint_root.iterdir()
        if candidate.is_dir()
        and model_id in candidate.name
        and (candidate / "checkpoint.pth").exists()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint match for model_id={model_id}, found {matches}"
        )
    return matches[0]


def make_decoder_input(
    batch_y: torch.Tensor, label_len: int, pred_len: int, device: torch.device
) -> torch.Tensor:
    """Create the decoder input without exposing future target values."""
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()
    return dec_inp.to(device)


def deterministic_context_corruption(
    exp: Exp_Long_Term_Forecast,
    batch_x: torch.Tensor,
    mode: str,
    batch_index: int,
    seed: int,
    dropout_p: float,
) -> torch.Tensor:
    """Apply deterministic context-only corruption so every model sees the same perturbation."""
    target, context = exp._split_target_context(batch_x)
    if context.size(-1) == 0:
        return batch_x

    corruption_seed = seed + (batch_index * 97) + CORRUPTION_MODE_OFFSET[mode]
    generator = torch.Generator(device=batch_x.device)
    generator.manual_seed(int(corruption_seed))

    if mode == "shuffle":
        channel_perm = torch.randperm(
            context.size(-1), generator=generator, device=context.device
        )
        corrupted_context = context[..., channel_perm]
    elif mode == "swap":
        if context.size(0) == 1:
            return batch_x
        batch_perm = torch.randperm(
            context.size(0), generator=generator, device=context.device
        )
        identity_perm = torch.arange(context.size(0), device=context.device)
        if torch.equal(batch_perm, identity_perm):
            batch_perm = torch.roll(batch_perm, shifts=1)
        corrupted_context = context[batch_perm]
    elif mode == "dropout":
        keep_prob = min(max(1.0 - dropout_p, 0.0), 1.0)
        if keep_prob == 0.0:
            corrupted_context = torch.zeros_like(context)
        else:
            keep_mask = (
                torch.rand(
                    context.shape,
                    generator=generator,
                    device=context.device,
                    dtype=context.dtype,
                )
                < keep_prob
            ).to(context.dtype)
            corrupted_context = context * keep_mask
    else:
        raise ValueError(f"Unsupported corruption mode: {mode}")
    return exp._merge_target_context(target, corrupted_context)


def safe_correlation(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    """Compute a stable Pearson correlation for flattened gate and utility values."""
    if len(values_x) < 2 or len(values_y) < 2:
        return 0.0
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    x_std = x.std()
    y_std = y.std()
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    return float(np.mean(centered_x * centered_y) / (x_std * y_std + 1e-8))


def init_utility_slice_store() -> Dict[str, List[float]]:
    """Initialize storage for utility-slice metrics."""
    return {
        "mse": [],
        "mae": [],
        "clean_gate": [],
        "clean_utility": [],
        "corrupt_gate": [],
        "corrupt_utility": [],
        "paired_gap": [],
        "paired_win": [],
    }


def init_horizon_store() -> Dict[str, List[float]]:
    """Initialize storage for horizon-bucket metrics."""
    return {
        "clean_sq": [],
        "clean_abs": [],
        "utility_drop": [],
        "gate_gap": [],
        "gate_mean": [],
    }


def init_corruption_store() -> Dict[str, List[float]]:
    """Initialize storage for corruption-specific metrics."""
    return {
        "utility_drop": [],
        "paired_gap": [],
        "paired_win": [],
        "metric_drop": [],
    }


def collect_reference_slice_labels(
    exp: Exp_Long_Term_Forecast, data_loader: Iterable, max_batches: int
) -> Dict[int, str]:
    """Use g0 clean sample-level utility terciles as the common slice definition."""
    sample_utilities: List[float] = []
    sample_index = 0
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if max_batches and batch_index >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = make_decoder_input(batch_y, exp.args.label_len, exp.args.pred_len, exp.device)
            outputs = exp._slice_outputs(exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
            target_y = exp._slice_target(batch_y)
            utility = (outputs["pred_h"] - target_y).pow(2) - (outputs["pred_c"] - target_y).pow(2)
            sample_util = utility.mean(dim=(1, 2)).detach().cpu().tolist()
            sample_utilities.extend(sample_util)
            sample_index += len(sample_util)

    lower, upper = np.quantile(np.asarray(sample_utilities, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    labels: Dict[int, str] = {}
    for index, utility_value in enumerate(sample_utilities):
        if utility_value <= lower:
            labels[index] = UTILITY_SLICE_NAMES[0]
        elif utility_value <= upper:
            labels[index] = UTILITY_SLICE_NAMES[1]
        else:
            labels[index] = UTILITY_SLICE_NAMES[2]
    return labels


def analyze_model(
    exp: Exp_Long_Term_Forecast,
    data_loader: Iterable,
    sample_slice_labels: Dict[int, str],
    max_batches: int,
    seed: int,
    dropout_p: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Collect utility-slice, horizon-slice, and corruption-type summaries for a model."""
    utility_store = {name: init_utility_slice_store() for name in UTILITY_SLICE_NAMES}
    horizon_store = {name: init_horizon_store() for name, _, _ in HORIZON_BUCKETS}
    corruption_store = {mode: init_corruption_store() for mode in CORRUPTION_MODES}

    sample_offset = 0
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if max_batches and batch_index >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = make_decoder_input(batch_y, exp.args.label_len, exp.args.pred_len, exp.device)
            clean_outputs = exp._slice_outputs(exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
            target_y = exp._slice_target(batch_y)

            clean_pred = clean_outputs["pred"]
            clean_pred_h = clean_outputs["pred_h"]
            clean_pred_c = clean_outputs["pred_c"]
            clean_gate = clean_outputs["gate"]
            clean_sq = (clean_pred - target_y).pow(2)
            clean_abs = (clean_pred - target_y).abs()
            clean_utility = (clean_pred_h - target_y).pow(2) - (clean_pred_c - target_y).pow(2)

            batch_size = target_y.size(0)
            sample_labels = [
                sample_slice_labels[sample_offset + sample_index]
                for sample_index in range(batch_size)
            ]
            sample_offset += batch_size

            for sample_index, slice_name in enumerate(sample_labels):
                store = utility_store[slice_name]
                store["mse"].append(float(clean_sq[sample_index].mean().item()))
                store["mae"].append(float(clean_abs[sample_index].mean().item()))
                store["clean_gate"].extend(clean_gate[sample_index].reshape(-1).detach().cpu().tolist())
                store["clean_utility"].extend(
                    clean_utility[sample_index].reshape(-1).detach().cpu().tolist()
                )

            for bucket_name, start, end in HORIZON_BUCKETS:
                horizon_store[bucket_name]["clean_sq"].extend(
                    clean_sq[:, start:end].reshape(-1).detach().cpu().tolist()
                )
                horizon_store[bucket_name]["clean_abs"].extend(
                    clean_abs[:, start:end].reshape(-1).detach().cpu().tolist()
                )
                horizon_store[bucket_name]["gate_mean"].append(
                    float(clean_gate[:, start:end].mean().item())
                )

            for mode in CORRUPTION_MODES:
                corrupt_batch_x = deterministic_context_corruption(
                    exp=exp,
                    batch_x=batch_x,
                    mode=mode,
                    batch_index=batch_index,
                    seed=seed,
                    dropout_p=dropout_p,
                )
                corrupt_outputs = exp._slice_outputs(
                    exp.model(corrupt_batch_x, batch_x_mark, dec_inp, batch_y_mark)
                )
                corrupt_pred = corrupt_outputs["pred"]
                corrupt_pred_h = corrupt_outputs["pred_h"]
                corrupt_pred_c = corrupt_outputs["pred_c"]
                corrupt_gate = corrupt_outputs["gate"]
                corrupt_sq = (corrupt_pred - target_y).pow(2)
                corrupt_utility = (corrupt_pred_h - target_y).pow(2) - (
                    corrupt_pred_c - target_y
                ).pow(2)

                corruption_store[mode]["utility_drop"].append(
                    float((clean_utility.mean() - corrupt_utility.mean()).item())
                )
                corruption_store[mode]["paired_gap"].append(
                    float((clean_gate.mean() - corrupt_gate.mean()).item())
                )
                corruption_store[mode]["paired_win"].append(
                    float((clean_gate > corrupt_gate).float().mean().item())
                )
                corruption_store[mode]["metric_drop"].append(
                    float((corrupt_sq.mean() - clean_sq.mean()).item())
                )

                for sample_index, slice_name in enumerate(sample_labels):
                    store = utility_store[slice_name]
                    store["corrupt_gate"].extend(
                        corrupt_gate[sample_index].reshape(-1).detach().cpu().tolist()
                    )
                    store["corrupt_utility"].extend(
                        corrupt_utility[sample_index].reshape(-1).detach().cpu().tolist()
                    )
                    store["paired_gap"].append(
                        float(
                            (
                                clean_gate[sample_index].mean()
                                - corrupt_gate[sample_index].mean()
                            ).item()
                        )
                    )
                    store["paired_win"].append(
                        float(
                            (
                                clean_gate[sample_index] > corrupt_gate[sample_index]
                            ).float().mean().item()
                        )
                    )

                for bucket_name, start, end in HORIZON_BUCKETS:
                    horizon_store[bucket_name]["utility_drop"].append(
                        float(
                            (
                                clean_utility[:, start:end].mean()
                                - corrupt_utility[:, start:end].mean()
                            ).item()
                        )
                    )
                    horizon_store[bucket_name]["gate_gap"].append(
                        float(
                            (
                                clean_gate[:, start:end].mean()
                                - corrupt_gate[:, start:end].mean()
                            ).item()
                        )
                    )

    utility_summary: Dict[str, Dict[str, float]] = {}
    for slice_name, store in utility_store.items():
        utility_summary[slice_name] = {
            "mse": float(np.mean(store["mse"])) if store["mse"] else 0.0,
            "mae": float(np.mean(store["mae"])) if store["mae"] else 0.0,
            "clean_gate_utility_corr": safe_correlation(
                store["clean_gate"], store["clean_utility"]
            ),
            "corrupt_gate_utility_corr": safe_correlation(
                store["corrupt_gate"], store["corrupt_utility"]
            ),
            "paired_gate_gap": float(np.mean(store["paired_gap"])) if store["paired_gap"] else 0.0,
            "paired_gate_win_rate": float(np.mean(store["paired_win"])) if store["paired_win"] else 0.0,
        }

    horizon_summary: Dict[str, Dict[str, float]] = {}
    for bucket_name, store in horizon_store.items():
        horizon_summary[bucket_name] = {
            "mse": float(np.mean(store["clean_sq"])) if store["clean_sq"] else 0.0,
            "mae": float(np.mean(store["clean_abs"])) if store["clean_abs"] else 0.0,
            "utility_drop": float(np.mean(store["utility_drop"])) if store["utility_drop"] else 0.0,
            "gate_gap": float(np.mean(store["gate_gap"])) if store["gate_gap"] else 0.0,
            "gate_mean": float(np.mean(store["gate_mean"])) if store["gate_mean"] else 0.0,
        }

    corruption_summary: Dict[str, Dict[str, float]] = {}
    for mode, store in corruption_store.items():
        corruption_summary[mode] = {
            "utility_drop": float(np.mean(store["utility_drop"])) if store["utility_drop"] else 0.0,
            "paired_gate_gap": float(np.mean(store["paired_gap"])) if store["paired_gap"] else 0.0,
            "paired_gate_win_rate": float(np.mean(store["paired_win"])) if store["paired_win"] else 0.0,
            "metric_drop": float(np.mean(store["metric_drop"])) if store["metric_drop"] else 0.0,
        }
    return utility_summary, horizon_summary, corruption_summary


def format_summary_table(summary: Dict[str, Dict[str, Dict[str, float]]], title: str) -> None:
    """Print a compact summary table for a nested metric dictionary."""
    print(f"\n[{title}]")
    for label, metrics_by_group in summary.items():
        print(f"{label}")
        for group_name, metrics in metrics_by_group.items():
            metric_parts = ", ".join(
                f"{key}={value:.4f}" for key, value in metrics.items()
            )
            print(f"  {group_name}: {metric_parts}")


def main() -> None:
    """Run slice analysis for the requested STIC checkpoints."""
    cli_args = parse_args()
    set_seed(cli_args.seed)
    model_specs = [parse_model_spec(spec) for spec in cli_args.model_spec]

    exp_map: Dict[str, Exp_Long_Term_Forecast] = {}
    checkpoint_map: Dict[str, str] = {}
    for model_spec in model_specs:
        exp_args = build_exp_args(cli_args, model_spec)
        exp = Exp_Long_Term_Forecast(exp_args)
        checkpoint_setting = resolve_checkpoint_setting(
            exp_args.checkpoints, model_spec.model_id
        )
        checkpoint_path = os.path.join(
            exp_args.checkpoints, checkpoint_setting, "checkpoint.pth"
        )
        exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))
        exp.model.eval()
        exp_map[model_spec.label] = exp
        checkpoint_map[model_spec.label] = checkpoint_setting

    reference_exp = exp_map["g0"]
    _, reference_loader = reference_exp._get_data("test")
    sample_slice_labels = collect_reference_slice_labels(
        reference_exp, reference_loader, cli_args.max_batches
    )

    results_utility: Dict[str, Dict[str, Dict[str, float]]] = {}
    results_horizon: Dict[str, Dict[str, Dict[str, float]]] = {}
    results_corruption: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_spec in model_specs:
        exp = exp_map[model_spec.label]
        _, test_loader = exp._get_data("test")
        utility_summary, horizon_summary, corruption_summary = analyze_model(
            exp=exp,
            data_loader=test_loader,
            sample_slice_labels=sample_slice_labels,
            max_batches=cli_args.max_batches,
            seed=cli_args.seed,
            dropout_p=cli_args.dropout_p,
        )
        results_utility[model_spec.label] = utility_summary
        results_horizon[model_spec.label] = horizon_summary
        results_corruption[model_spec.label] = corruption_summary

    print("Resolved checkpoints:")
    for label, setting in checkpoint_map.items():
        print(f"  {label}: {setting}")
    format_summary_table(results_utility, "utility_slices")
    format_summary_table(results_horizon, "horizon_slices")
    format_summary_table(results_corruption, "corruption_types")


if __name__ == "__main__":
    main()
