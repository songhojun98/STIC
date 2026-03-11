from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp


class _DLinearBranch(nn.Module):
    """DLinear-style forecasting branch used inside STIC."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        moving_avg: int,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.decomposition = series_decomp(moving_avg)

        if individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(channels):
                seasonal = nn.Linear(seq_len, pred_len)
                trend = nn.Linear(seq_len, pred_len)
                seasonal.weight = nn.Parameter(
                    (1.0 / seq_len) * torch.ones(pred_len, seq_len)
                )
                trend.weight = nn.Parameter(
                    (1.0 / seq_len) * torch.ones(pred_len, seq_len)
                )
                self.linear_seasonal.append(seasonal)
                self.linear_trend.append(trend)
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_seasonal.weight = nn.Parameter(
                (1.0 / seq_len) * torch.ones(pred_len, seq_len)
            )
            self.linear_trend.weight = nn.Parameter(
                (1.0 / seq_len) * torch.ones(pred_len, seq_len)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from historical inputs shaped `[B, L, C]`."""
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0),
                seasonal_init.size(1),
                self.pred_len,
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                trend_init.size(0),
                trend_init.size(1),
                self.pred_len,
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for channel_index in range(self.channels):
                seasonal_output[:, channel_index, :] = self.linear_seasonal[channel_index](
                    seasonal_init[:, channel_index, :]
                )
                trend_output[:, channel_index, :] = self.linear_trend[channel_index](
                    trend_init[:, channel_index, :]
                )
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        return (seasonal_output + trend_output).permute(0, 2, 1)


class _TargetSpecificContextBranch(nn.Module):
    """Mix all channels into a target-specific sequence before the temporal head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_channels: int,
        moving_avg: int,
        target_index: int,
        mixer_type: str = "linear",
        mixer_hidden_dim: int = 0,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.target_index = target_index
        self.residual_scale = float(residual_scale)
        resolved_hidden_dim = mixer_hidden_dim or max(8, min(64, input_channels * 4))

        if mixer_type == "linear":
            self.channel_mixer = nn.Linear(input_channels, 1)
        elif mixer_type == "mlp":
            self.channel_mixer = nn.Sequential(
                nn.Linear(input_channels, resolved_hidden_dim),
                nn.GELU(),
                nn.Linear(resolved_hidden_dim, 1),
            )
        else:
            raise ValueError(
                "Unsupported STIC context mixer. Expected one of: linear, mlp."
            )

        self.temporal_head = _DLinearBranch(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=1,
            moving_avg=moving_avg,
            individual=False,
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return prediction plus intermediate tensors for the context-aware branch."""
        target_history = x[..., self.target_index : self.target_index + 1]
        mixed_residual = self.residual_scale * self.channel_mixer(x)
        mixed_sequence = target_history + mixed_residual
        pred = self.temporal_head(mixed_sequence)
        return {
            "pred": pred,
            "target_history": target_history,
            "mixed_residual": mixed_residual,
            "mixed_sequence": mixed_sequence,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast a target-specific sequence from `[B, L, D]` multivariate inputs."""
        return self.encode(x)["pred"]


class Model(nn.Module):
    """Selective Trust in Context with a target-only history branch and mixed context branch."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = (
            configs.seq_len
            if self.task_name in {"classification", "anomaly_detection", "imputation"}
            else configs.pred_len
        )
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, "individual", False)
        self.target_index = self._resolve_target_index(configs)
        self.stic_mode = getattr(configs, "stic_mode", "dynamic").lower()
        self.gate_input_mode = getattr(configs, "stic_gate_input_mode", "g0").lower()
        self.gate_hidden_feat_dim = max(
            1, int(getattr(configs, "stic_gate_hidden_feat_dim", 8))
        )
        self.gate_stats_mode = getattr(configs, "stic_gate_stats_mode", "basic").lower()
        self.gate_std_scale = float(getattr(configs, "stic_gate_std_scale", 1.0))
        self.gate_hidden_scale = float(
            getattr(configs, "stic_gate_hidden_scale", 1.0)
        )
        self.gate_summary_reg_mode = getattr(
            configs, "stic_gate_summary_reg_mode", "none"
        ).lower()
        self.gate_summary_clip_value = float(
            getattr(configs, "stic_gate_summary_clip_value", 1.0)
        )
        if self.gate_input_mode == "g1b-sumreg-rms":
            self.gate_summary_reg_mode = "rms"
        elif self.gate_input_mode == "g1b-sumreg-clip":
            self.gate_summary_reg_mode = "clip"
        self.static_gate_value = float(
            min(1.0, max(0.0, getattr(configs, "stic_static_gate_value", 0.5)))
        )

        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                "STIC currently supports only long-term and short-term forecasting tasks."
            )
        if self.stic_mode not in {
            "dynamic",
            "static",
            "history_only",
            "always_on",
            "no_gate",
        }:
            raise ValueError(
                "Unsupported STIC mode. Expected one of: "
                "dynamic, static, history_only, always_on, no_gate."
            )
        if self.gate_input_mode not in {
            "g0",
            "g1",
            "g1a",
            "g1b",
            "g1c",
            "g1-lite",
            "g1-diff",
            "g1-norm",
            "g1b-meanheavy",
            "g1b-diff-lite",
            "g1b-topclip",
            "g1b-topclip-lite",
            "g1b-sumreg-rms",
            "g1b-sumreg-clip",
            "g2",
        }:
            raise ValueError(
                "Unsupported STIC gate input mode. Expected one of: "
                "g0, g1, g1a, g1b, g1c, g1-lite, g1-diff, g1-norm, "
                "g1b-meanheavy, g1b-diff-lite, g1b-topclip, g1b-topclip-lite, "
                "g1b-sumreg-rms, g1b-sumreg-clip, g2."
            )
        if self.gate_stats_mode not in {"basic"}:
            raise ValueError("Unsupported STIC gate stats mode. Expected: basic.")
        if self.gate_summary_reg_mode not in {"none", "rms", "clip"}:
            raise ValueError(
                "Unsupported STIC gate summary regularization mode. Expected: none, rms, clip."
            )

        self.history_branch = _DLinearBranch(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=1,
            moving_avg=configs.moving_avg,
            individual=False,
        )
        self.context_branch = _TargetSpecificContextBranch(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            moving_avg=configs.moving_avg,
            input_channels=self.enc_in,
            target_index=self.target_index,
            mixer_type=getattr(configs, "stic_context_mixer_type", "linear"),
            mixer_hidden_dim=getattr(configs, "stic_context_mixer_hidden_dim", 0),
            residual_scale=getattr(configs, "stic_context_residual_scale", 0.5),
        )
        gate_input_dim = self._resolve_gate_input_dim()
        gate_hidden = max(16, min(128, self.pred_len))
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _resolve_target_index(configs) -> int:
        """Resolve the target channel index without using future information."""
        if getattr(configs, "features", "S") == "S":
            return 0
        configured_target_index = getattr(configs, "stic_target_index", -1)
        if configured_target_index >= 0:
            return min(int(configured_target_index), max(int(configs.enc_in) - 1, 0))
        return max(int(configs.enc_in) - 1, 0)

    def _select_target_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Select target history shaped `[B, L, 1]`."""
        return x[..., self.target_index : self.target_index + 1]

    def _select_context_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Select non-target context channels shaped `[B, L, D-1]`."""
        if x.size(-1) <= 1:
            return x.new_zeros(x.size(0), x.size(1), 0)
        left = x[..., : self.target_index]
        right = x[..., self.target_index + 1 :]
        return torch.cat((left, right), dim=-1)

    def _resolve_gate_input_dim(self) -> int:
        """Return the final feature width consumed by the gate MLP."""
        if self.gate_input_mode == "g0":
            return 3
        if self.gate_input_mode in {"g1a", "g1c"}:
            return 3 + (2 * self.gate_hidden_feat_dim) + (1 if self.gate_input_mode == "g1c" else 0)
        if self.gate_input_mode == "g1-lite":
            return 3 + (2 * self.gate_hidden_feat_dim)
        if self.gate_input_mode in {"g1-diff", "g1b-diff-lite"}:
            return 3 + (2 * self.gate_hidden_feat_dim)
        if self.gate_input_mode == "g1-norm":
            return 3 + (4 * self.gate_hidden_feat_dim)
        if self.gate_input_mode in {
            "g1",
            "g1b",
            "g1b-meanheavy",
            "g1b-topclip",
            "g1b-topclip-lite",
            "g1b-sumreg-rms",
            "g1b-sumreg-clip",
        }:
            return 3 + (4 * self.gate_hidden_feat_dim)
        return 3 + (4 * self.gate_hidden_feat_dim) + 4

    def _pool_branch_feature(
        self, sequence: torch.Tensor, include_std: bool, std_scale: float = 1.0
    ) -> torch.Tensor:
        """Pool `[B, L, 1]` branch features into a fixed-width summary."""
        sequence_bt = sequence.transpose(1, 2)
        pooled_mean = F.adaptive_avg_pool1d(
            sequence_bt, output_size=self.gate_hidden_feat_dim
        ).squeeze(1)
        if not include_std:
            return pooled_mean
        pooled_second_moment = F.adaptive_avg_pool1d(
            sequence_bt.pow(2), output_size=self.gate_hidden_feat_dim
        ).squeeze(1)
        pooled_var = (pooled_second_moment - pooled_mean.pow(2)).clamp_min(0.0)
        pooled_std = std_scale * torch.sqrt(pooled_var + 1e-6)
        return torch.cat((pooled_mean, pooled_std), dim=-1)

    def _expand_gate_summary(self, summary: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-sample gate summary `[B, D]` across the forecast horizon."""
        return summary.unsqueeze(1).expand(-1, self.pred_len, -1)

    @staticmethod
    def _normalize_gate_summary(summary: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of a gate summary for scale-stable comparison."""
        return F.layer_norm(summary, normalized_shape=(summary.size(-1),))

    @staticmethod
    def _rms_regularize_gate_summary(summary: torch.Tensor) -> torch.Tensor:
        """Shrink only high-RMS summaries while keeping low-energy summaries intact."""
        rms = torch.sqrt(summary.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        scale = torch.clamp(1.0 / rms, max=1.0)
        return summary * scale

    def _clip_regularize_gate_summary(self, summary: torch.Tensor) -> torch.Tensor:
        """Apply max-norm clipping to per-horizon summary vectors."""
        summary_norm = torch.norm(summary, dim=-1, keepdim=True)
        scale = torch.clamp(
            self.gate_summary_clip_value / (summary_norm + 1e-6), max=1.0
        )
        return summary * scale

    def _regularize_gate_summary(self, summary: torch.Tensor) -> torch.Tensor:
        """Apply the configured summary regularization to hidden gate features only."""
        if self.gate_summary_reg_mode == "rms":
            return self._rms_regularize_gate_summary(summary)
        if self.gate_summary_reg_mode == "clip":
            return self._clip_regularize_gate_summary(summary)
        return summary

    def _build_gate_stats(
        self,
        x_enc: torch.Tensor,
        target_history: torch.Tensor,
        mixed_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Build compact per-sample statistics for the richer gate input."""
        context = self._select_context_channels(x_enc)
        if context.size(-1) == 0:
            ctx_mean = target_history.new_zeros(target_history.size(0), 1)
            ctx_std = target_history.new_zeros(target_history.size(0), 1)
        else:
            ctx_mean = context.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            ctx_std = context.std(dim=(1, 2), unbiased=False, keepdim=False).unsqueeze(-1)
        mixed_residual_abs_mean = mixed_residual.abs().mean(dim=(1, 2)).unsqueeze(-1)
        target_hist_std = target_history.std(dim=1, unbiased=False).squeeze(-1).unsqueeze(-1)
        stats = torch.cat(
            (ctx_mean, ctx_std, mixed_residual_abs_mean, target_hist_std), dim=-1
        )
        return stats.unsqueeze(1).expand(-1, self.pred_len, -1)

    @staticmethod
    def build_gate_input_g0(pred_h: torch.Tensor, pred_c: torch.Tensor) -> torch.Tensor:
        """Build the baseline prediction-level gate input `[B, H, 3]`."""
        return torch.cat((pred_h, pred_c, pred_c - pred_h), dim=-1)

    def build_gate_input_g1a(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a mean-pooled representation gate input."""
        return torch.cat(
            (gate_feat_h, gate_feat_c, self.build_gate_input_g0(pred_h, pred_c)), dim=-1
        )

    def build_gate_input_g1b(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a mean-plus-std pooled representation gate input."""
        return torch.cat(
            (gate_feat_h, gate_feat_c, self.build_gate_input_g0(pred_h, pred_c)), dim=-1
        )

    def build_gate_input_g1b_meanheavy(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a mean-heavy G1B gate input with down-weighted std summaries."""
        return torch.cat(
            (gate_feat_h, gate_feat_c, self.build_gate_input_g0(pred_h, pred_c)), dim=-1
        )

    def build_gate_input_g1b_diff_lite(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a compact G1B refinement using only branch-summary differences."""
        return torch.cat(
            (gate_feat_c - gate_feat_h, self.build_gate_input_g0(pred_h, pred_c)),
            dim=-1,
        )

    def build_gate_input_g1b_topclip(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a clipped G1B gate input by shrinking hidden-summary influence."""
        scaled_h = self.gate_hidden_scale * gate_feat_h
        scaled_c = self.gate_hidden_scale * gate_feat_c
        return torch.cat(
            (scaled_h, scaled_c, self.build_gate_input_g0(pred_h, pred_c)), dim=-1
        )

    def build_gate_input_g1b_topclip_lite(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a milder topclip variant using a lighter hidden-summary scale."""
        scaled_h = self.gate_hidden_scale * gate_feat_h
        scaled_c = self.gate_hidden_scale * gate_feat_c
        return torch.cat(
            (scaled_h, scaled_c, self.build_gate_input_g0(pred_h, pred_c)), dim=-1
        )

    def build_gate_input_g1b_sumreg(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a G1B variant with lightweight hidden-summary regularization."""
        regularized_h = self._regularize_gate_summary(gate_feat_h)
        regularized_c = self._regularize_gate_summary(gate_feat_c)
        return torch.cat(
            (regularized_h, regularized_c, self.build_gate_input_g0(pred_h, pred_c)),
            dim=-1,
        )

    def build_gate_input_g1c(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a mean-pooled representation gate input with an absolute discrepancy feature."""
        pred_delta_abs = (pred_c - pred_h).abs()
        return torch.cat(
            (
                gate_feat_h,
                gate_feat_c,
                self.build_gate_input_g0(pred_h, pred_c),
                pred_delta_abs,
            ),
            dim=-1,
        )

    def build_gate_input_g1_lite(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a compact gate input that keeps only the context-branch summary."""
        return torch.cat(
            (gate_feat_c, self.build_gate_input_g0(pred_h, pred_c)),
            dim=-1,
        )

    def build_gate_input_g1_diff(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a compact gate input centered on branch-summary discrepancy."""
        gate_feat_diff = gate_feat_c - gate_feat_h
        return torch.cat(
            (gate_feat_diff, self.build_gate_input_g0(pred_h, pred_c)),
            dim=-1,
        )

    def build_gate_input_g1_norm(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """Build a compact gate input using normalized branch summaries."""
        normalized_h = self._normalize_gate_summary(gate_feat_h)
        normalized_c = self._normalize_gate_summary(gate_feat_c)
        return torch.cat(
            (normalized_h, normalized_c, self.build_gate_input_g0(pred_h, pred_c)),
            dim=-1,
        )

    def build_gate_input_g2(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: torch.Tensor,
        gate_feat_c: torch.Tensor,
        gate_stats: torch.Tensor,
    ) -> torch.Tensor:
        """Build the richer gate input with summaries and simple statistics."""
        return torch.cat(
            (
                gate_feat_h,
                gate_feat_c,
                self.build_gate_input_g0(pred_h, pred_c),
                gate_stats,
            ),
            dim=-1,
        )

    def _build_gate_input(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h: Optional[torch.Tensor] = None,
        gate_feat_c: Optional[torch.Tensor] = None,
        gate_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select the configured gate input builder."""
        if self.gate_input_mode == "g0":
            return self.build_gate_input_g0(pred_h, pred_c)
        if gate_feat_h is None or gate_feat_c is None:
            raise ValueError("G1/G2 gate modes require branch summaries.")
        if self.gate_input_mode == "g1a":
            return self.build_gate_input_g1a(pred_h, pred_c, gate_feat_h, gate_feat_c)
        if self.gate_input_mode in {"g1", "g1b"}:
            return self.build_gate_input_g1b(pred_h, pred_c, gate_feat_h, gate_feat_c)
        if self.gate_input_mode == "g1b-meanheavy":
            return self.build_gate_input_g1b_meanheavy(
                pred_h, pred_c, gate_feat_h, gate_feat_c
            )
        if self.gate_input_mode == "g1b-diff-lite":
            return self.build_gate_input_g1b_diff_lite(
                pred_h, pred_c, gate_feat_h, gate_feat_c
            )
        if self.gate_input_mode == "g1b-topclip":
            return self.build_gate_input_g1b_topclip(
                pred_h, pred_c, gate_feat_h, gate_feat_c
            )
        if self.gate_input_mode == "g1b-topclip-lite":
            return self.build_gate_input_g1b_topclip_lite(
                pred_h, pred_c, gate_feat_h, gate_feat_c
            )
        if self.gate_input_mode in {"g1b-sumreg-rms", "g1b-sumreg-clip"}:
            return self.build_gate_input_g1b_sumreg(
                pred_h, pred_c, gate_feat_h, gate_feat_c
            )
        if self.gate_input_mode == "g1c":
            return self.build_gate_input_g1c(pred_h, pred_c, gate_feat_h, gate_feat_c)
        if self.gate_input_mode == "g1-lite":
            return self.build_gate_input_g1_lite(pred_h, pred_c, gate_feat_c)
        if self.gate_input_mode == "g1-diff":
            return self.build_gate_input_g1_diff(
                pred_h, pred_c, gate_feat_h, gate_feat_c
            )
        if self.gate_input_mode == "g1-norm":
            return self.build_gate_input_g1_norm(
                pred_h, pred_c, gate_feat_h, gate_feat_c
            )
        if gate_stats is None:
            raise ValueError("G2 gate mode requires basic statistics features.")
        return self.build_gate_input_g2(
            pred_h, pred_c, gate_feat_h, gate_feat_c, gate_stats
        )

    @staticmethod
    def _constant_gate(reference: torch.Tensor, value: float) -> torch.Tensor:
        """Create a constant gate tensor with the same shape as `reference`."""
        return torch.full_like(reference, fill_value=value)

    def _apply_mode(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        has_context: bool,
        gate_feat_h: Optional[torch.Tensor] = None,
        gate_feat_c: Optional[torch.Tensor] = None,
        gate_stats: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Combine branch outputs according to the selected STIC ablation mode."""
        if not has_context or self.stic_mode == "history_only":
            gate = self._constant_gate(pred_h, 0.0)
            pred = pred_h
            return {
                "pred": pred,
                "gate": gate,
                "gate_trainable": False,
                "aux_loss_enabled": False,
                "mode": "history_only" if self.stic_mode == "history_only" else "no_context",
            }

        if self.stic_mode in {"always_on", "no_gate"}:
            gate = self._constant_gate(pred_h, 1.0)
            pred = pred_c
            return {
                "pred": pred,
                "gate": gate,
                "gate_trainable": False,
                "aux_loss_enabled": True,
                "mode": "always_on" if self.stic_mode == "always_on" else "no_gate",
            }

        if self.stic_mode == "static":
            gate = self._constant_gate(pred_h, self.static_gate_value)
            pred = pred_h + gate * (pred_c - pred_h)
            return {
                "pred": pred,
                "gate": gate,
                "gate_trainable": False,
                "aux_loss_enabled": True,
                "mode": "static",
            }

        extra_outputs: Dict[str, Any] = {}
        gate_input = self._build_gate_input(
            pred_h=pred_h,
            pred_c=pred_c,
            gate_feat_h=gate_feat_h,
            gate_feat_c=gate_feat_c,
            gate_stats=gate_stats,
        )
        gate = self.gate_network(gate_input)
        extra_outputs["gate_input"] = gate_input
        pred = pred_h + gate * (pred_c - pred_h)
        outputs = {
            "pred": pred,
            "gate": gate,
            "gate_trainable": True,
            "aux_loss_enabled": True,
            "mode": "dynamic",
        }
        outputs.update(extra_outputs)
        return outputs

    def forecast(self, x_enc: torch.Tensor) -> Dict[str, Any]:
        """Run STIC forecasting and return branch outputs plus gate."""
        target_history = self._select_target_channel(x_enc)
        pred_h = self.history_branch(target_history)
        has_context = x_enc.size(-1) > 1
        gate_feat_h_raw = target_history
        gate_feat_h = None
        gate_feat_c_raw = None
        gate_feat_c = None
        gate_stats = None

        if not has_context or self.stic_mode == "history_only":
            pred_c = pred_h
            mixed_residual = target_history.new_zeros(target_history.shape)
            mixed_sequence = target_history
        else:
            context_outputs = self.context_branch.encode(x_enc)
            pred_c = context_outputs["pred"]
            mixed_residual = context_outputs["mixed_residual"]
            mixed_sequence = context_outputs["mixed_sequence"]
        gate_feat_c_raw = mixed_sequence

        if self.gate_input_mode in {
            "g1a",
            "g1b",
            "g1c",
            "g1",
            "g1-lite",
            "g1-diff",
            "g1-norm",
            "g1b-meanheavy",
            "g1b-diff-lite",
            "g1b-topclip",
            "g1b-topclip-lite",
            "g1b-sumreg-rms",
            "g1b-sumreg-clip",
            "g2",
        }:
            include_std = self.gate_input_mode in {
                "g1",
                "g1b",
                "g1-lite",
                "g1-diff",
                "g1-norm",
                "g1b-meanheavy",
                "g1b-diff-lite",
                "g1b-topclip",
                "g1b-topclip-lite",
                "g1b-sumreg-rms",
                "g1b-sumreg-clip",
                "g2",
            }
            std_scale = self.gate_std_scale if self.gate_input_mode == "g1b-meanheavy" else 1.0
            gate_feat_h = self._expand_gate_summary(
                self._pool_branch_feature(
                    target_history,
                    include_std=include_std,
                    std_scale=std_scale,
                )
            )
            gate_feat_c = self._expand_gate_summary(
                self._pool_branch_feature(
                    mixed_sequence,
                    include_std=include_std,
                    std_scale=std_scale,
                )
            )
        if self.gate_input_mode == "g2":
            gate_stats = self._build_gate_stats(
                x_enc=x_enc,
                target_history=target_history,
                mixed_residual=mixed_residual,
            )

        outputs = self._apply_mode(
            pred_h=pred_h,
            pred_c=pred_c,
            has_context=has_context,
            gate_feat_h=gate_feat_h,
            gate_feat_c=gate_feat_c,
            gate_stats=gate_stats,
        )
        outputs["pred_h"] = pred_h
        outputs["pred_c"] = pred_c
        outputs["gate_input_mode"] = self.gate_input_mode
        outputs["gate_feat_h_raw"] = gate_feat_h_raw
        outputs["gate_feat_c_raw"] = gate_feat_c_raw
        if gate_feat_h is not None:
            outputs["gate_feat_h"] = gate_feat_h
        if gate_feat_c is not None:
            outputs["gate_feat_c"] = gate_feat_c
        if gate_stats is not None:
            outputs["gate_stats"] = gate_stats
        return outputs

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        x_mark_dec: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.forecast(x_enc)
