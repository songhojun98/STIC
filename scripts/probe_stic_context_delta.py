import argparse
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe how much pred_c changes between clean and corrupted context."
    )
    parser.add_argument("--checkpoint_setting", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument(
        "--probe_modes",
        type=str,
        nargs="+",
        default=["shuffle", "swap", "dropout"],
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
    parser.add_argument(
        "--stic_context_mixer_type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
    )
    parser.add_argument("--stic_context_mixer_hidden_dim", type=int, default=0)
    parser.add_argument("--stic_context_residual_scale", type=float, default=0.5)
    return parser.parse_args()


def build_exp_args(cli_args):
    use_gpu = torch.cuda.is_available() and not cli_args.cpu
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=0,
        model_id="probe",
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
        des="probe",
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
        stic_target_index=cli_args.stic_target_index,
        stic_context_mixer_type=cli_args.stic_context_mixer_type,
        stic_context_mixer_hidden_dim=cli_args.stic_context_mixer_hidden_dim,
        stic_context_residual_scale=cli_args.stic_context_residual_scale,
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_decoder_input(batch_y, label_len, pred_len, device):
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()
    return dec_inp.to(device)


def summarize(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def compute_probe_metrics(clean_outputs, corrupt_outputs, target_y, clean_batch_x, corrupt_batch_x, exp):
    clean_pred = clean_outputs["pred"]
    clean_pred_h = clean_outputs["pred_h"]
    clean_pred_c = clean_outputs["pred_c"]
    clean_gate = clean_outputs["gate"]

    corrupt_pred = corrupt_outputs["pred"]
    corrupt_pred_h = corrupt_outputs["pred_h"]
    corrupt_pred_c = corrupt_outputs["pred_c"]
    corrupt_gate = corrupt_outputs["gate"]

    clean_target_x, clean_context_x = exp._split_target_context(clean_batch_x)
    corrupt_target_x, corrupt_context_x = exp._split_target_context(corrupt_batch_x)

    clean_err_c = (clean_pred_c - target_y).pow(2)
    corrupt_err_c = (corrupt_pred_c - target_y).pow(2)
    clean_err_h = (clean_pred_h - target_y).pow(2)
    corrupt_err_h = (corrupt_pred_h - target_y).pow(2)
    clean_utility = clean_err_h - clean_err_c
    corrupt_utility = corrupt_err_h - corrupt_err_c

    pred_c_diff = corrupt_pred_c - clean_pred_c
    pred_diff = corrupt_pred - clean_pred
    gate_diff = corrupt_gate - clean_gate

    pred_c_rel = pred_c_diff.pow(2).mean().sqrt() / (clean_pred_c.pow(2).mean().sqrt() + 1e-8)
    pred_rel = pred_diff.pow(2).mean().sqrt() / (clean_pred.pow(2).mean().sqrt() + 1e-8)

    return {
        "context_input_abs_delta": (corrupt_context_x - clean_context_x).abs().mean().item(),
        "target_input_abs_delta": (corrupt_target_x - clean_target_x).abs().mean().item(),
        "pred_c_abs_delta": pred_c_diff.abs().mean().item(),
        "pred_c_rmse_delta": pred_c_diff.pow(2).mean().sqrt().item(),
        "pred_c_rel_rmse_delta": pred_c_rel.item(),
        "pred_abs_delta": pred_diff.abs().mean().item(),
        "pred_rel_rmse_delta": pred_rel.item(),
        "pred_h_abs_delta": (corrupt_pred_h - clean_pred_h).abs().mean().item(),
        "gate_abs_delta": gate_diff.abs().mean().item(),
        "gate_mean_clean": clean_gate.mean().item(),
        "gate_mean_corrupt": corrupt_gate.mean().item(),
        "utility_mean_clean": clean_utility.mean().item(),
        "utility_mean_corrupt": corrupt_utility.mean().item(),
        "utility_drop": (clean_utility.mean() - corrupt_utility.mean()).item(),
        "utility_sign_flip_rate": (
            (clean_utility > 0) != (corrupt_utility > 0)
        ).float().mean().item(),
        "context_better_rate_clean": (clean_utility > 0).float().mean().item(),
        "context_better_rate_corrupt": (corrupt_utility > 0).float().mean().item(),
    }


def main():
    cli_args = parse_args()
    set_seed(cli_args.seed)
    exp_args = build_exp_args(cli_args)

    exp = Exp_Long_Term_Forecast(exp_args)
    checkpoint_path = os.path.join(
        exp_args.checkpoints, cli_args.checkpoint_setting, "checkpoint.pth"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))
    exp.model.eval()

    _, loader = exp._get_data(flag=cli_args.split)
    results = {mode: defaultdict(list) for mode in cli_args.probe_modes}
    first_batch_examples = {}

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if batch_index >= cli_args.max_batches:
                break

            clean_batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = make_decoder_input(batch_y, exp.args.label_len, exp.args.pred_len, exp.device)
            target_y = exp._slice_target(batch_y)

            clean_outputs = exp.model(clean_batch_x, batch_x_mark, dec_inp, batch_y_mark)
            clean_outputs = exp._slice_outputs(clean_outputs)

            for mode in cli_args.probe_modes:
                exp.args.stic_context_corruption_mode = mode
                exp.args.stic_context_corruption_prob = 1.0
                corrupt_batch_x, corrupted_batch, applied_mode = exp._apply_context_corruption(
                    clean_batch_x.clone()
                )
                if not corrupted_batch:
                    continue
                corrupt_outputs = exp.model(
                    corrupt_batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                corrupt_outputs = exp._slice_outputs(corrupt_outputs)

                metrics = compute_probe_metrics(
                    clean_outputs,
                    corrupt_outputs,
                    target_y,
                    clean_batch_x,
                    corrupt_batch_x,
                    exp,
                )
                metrics["applied_mode"] = applied_mode
                for key, value in metrics.items():
                    if key == "applied_mode":
                        continue
                    results[mode][key].append(value)
                if batch_index == 0:
                    first_batch_examples[mode] = metrics

    print(f"checkpoint_setting={cli_args.checkpoint_setting}")
    print(f"split={cli_args.split} max_batches={cli_args.max_batches}")
    for mode in cli_args.probe_modes:
        mode_results = results[mode]
        if not mode_results:
            print(f"[{mode}] no corrupted batches produced")
            continue
        print(f"[{mode}]")
        print(
            "  input_context_abs_delta={:.6f} target_input_abs_delta={:.6f}".format(
                summarize(mode_results["context_input_abs_delta"]),
                summarize(mode_results["target_input_abs_delta"]),
            )
        )
        print(
            "  pred_c_abs_delta={:.6f} pred_c_rmse_delta={:.6f} pred_c_rel_rmse_delta={:.6f}".format(
                summarize(mode_results["pred_c_abs_delta"]),
                summarize(mode_results["pred_c_rmse_delta"]),
                summarize(mode_results["pred_c_rel_rmse_delta"]),
            )
        )
        print(
            "  pred_abs_delta={:.6f} pred_rel_rmse_delta={:.6f} pred_h_abs_delta={:.6f}".format(
                summarize(mode_results["pred_abs_delta"]),
                summarize(mode_results["pred_rel_rmse_delta"]),
                summarize(mode_results["pred_h_abs_delta"]),
            )
        )
        print(
            "  gate_mean_clean={:.6f} gate_mean_corrupt={:.6f} gate_abs_delta={:.6f}".format(
                summarize(mode_results["gate_mean_clean"]),
                summarize(mode_results["gate_mean_corrupt"]),
                summarize(mode_results["gate_abs_delta"]),
            )
        )
        print(
            "  utility_mean_clean={:.6f} utility_mean_corrupt={:.6f} utility_drop={:.6f} utility_sign_flip_rate={:.6f}".format(
                summarize(mode_results["utility_mean_clean"]),
                summarize(mode_results["utility_mean_corrupt"]),
                summarize(mode_results["utility_drop"]),
                summarize(mode_results["utility_sign_flip_rate"]),
            )
        )
        print(
            "  context_better_rate_clean={:.6f} context_better_rate_corrupt={:.6f}".format(
                summarize(mode_results["context_better_rate_clean"]),
                summarize(mode_results["context_better_rate_corrupt"]),
            )
        )
        if mode in first_batch_examples:
            example = first_batch_examples[mode]
            print(
                "  first_batch_example: pred_c_abs_delta={:.6f}, gate_abs_delta={:.6f}, utility_drop={:.6f}".format(
                    example["pred_c_abs_delta"],
                    example["gate_abs_delta"],
                    example["utility_drop"],
                )
            )


if __name__ == "__main__":
    main()
