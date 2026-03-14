from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def clone_configs(configs: Any, **overrides: Any) -> Any:
    """Deep-copy an argparse namespace-like config and override selected fields."""
    cloned = copy.deepcopy(configs)
    for name, value in overrides.items():
        setattr(cloned, name, value)
    return cloned


def resolve_target_index(configs: Any) -> int:
    """Resolve the target channel index without using future information."""
    if getattr(configs, "features", "S") == "S":
        return 0
    configured_target_index = getattr(configs, "stic_target_index", -1)
    if configured_target_index >= 0:
        return min(int(configured_target_index), max(int(configs.enc_in) - 1, 0))
    return max(int(configs.enc_in) - 1, 0)


def select_target_channel(x: torch.Tensor, target_index: int) -> torch.Tensor:
    """Select the target channel as `[B, L, 1]`."""
    return x[..., target_index : target_index + 1]


def move_target_last(x: torch.Tensor, target_index: int) -> torch.Tensor:
    """Move the target channel to the final slot to keep a consistent convention."""
    if x.size(-1) <= 1 or target_index == x.size(-1) - 1:
        return x
    target = x[..., target_index : target_index + 1]
    context = torch.cat((x[..., :target_index], x[..., target_index + 1 :]), dim=-1)
    return torch.cat((context, target), dim=-1)


class TargetSequenceMixer(nn.Module):
    """Mix multivariate inputs into a target-specific residual sequence."""

    def __init__(
        self,
        input_channels: int,
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
                "Unsupported exogenous context mixer. Expected one of: linear, mlp."
            )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_history = x[..., self.target_index : self.target_index + 1]
        mixed_residual = self.residual_scale * self.channel_mixer(x)
        mixed_sequence = target_history + mixed_residual
        return {
            "target_history": target_history,
            "mixed_residual": mixed_residual,
            "mixed_sequence": mixed_sequence,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)["mixed_sequence"]


class SelectiveTrustGate(nn.Module):
    """Shared STIC-style trust gate for the exogenous backbone variants."""

    def __init__(self, configs: Any, pred_len: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.stic_mode = getattr(configs, "stic_mode", "dynamic").lower()
        self.gate_input_mode = getattr(configs, "stic_gate_input_mode", "g0").lower()
        self.gate_hidden_feat_dim = max(
            1, int(getattr(configs, "stic_gate_hidden_feat_dim", 8))
        )
        self.gate_hidden_scale = float(
            getattr(configs, "stic_gate_hidden_scale", 1.0)
        )
        self.static_gate_value = float(
            min(1.0, max(0.0, getattr(configs, "stic_static_gate_value", 0.5)))
        )

        valid_modes = {
            "dynamic",
            "static",
            "history_only",
            "always_on",
            "no_gate",
        }
        if self.stic_mode not in valid_modes:
            raise ValueError(
                "Unsupported STIC mode. Expected one of: "
                "dynamic, static, history_only, always_on, no_gate."
            )

        valid_gate_inputs = {"g0", "g1", "g1b", "g1b-topclip", "g1b-topclip-lite"}
        if self.gate_input_mode not in valid_gate_inputs:
            raise ValueError(
                "Unsupported exogenous STIC gate input mode. Expected one of: "
                "g0, g1, g1b, g1b-topclip, g1b-topclip-lite."
            )

        gate_input_dim = 3 if self.gate_input_mode == "g0" else 3 + (
            4 * self.gate_hidden_feat_dim
        )
        gate_hidden = max(16, min(128, self.pred_len))
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    def _pool_branch_feature(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence_bt = sequence.transpose(1, 2)
        pooled_mean = F.adaptive_avg_pool1d(
            sequence_bt, output_size=self.gate_hidden_feat_dim
        ).squeeze(1)
        pooled_second_moment = F.adaptive_avg_pool1d(
            sequence_bt.pow(2), output_size=self.gate_hidden_feat_dim
        ).squeeze(1)
        pooled_var = (pooled_second_moment - pooled_mean.pow(2)).clamp_min(0.0)
        pooled_std = torch.sqrt(pooled_var + 1e-6)
        return torch.cat((pooled_mean, pooled_std), dim=-1)

    def _expand_gate_summary(self, summary: torch.Tensor) -> torch.Tensor:
        return summary.unsqueeze(1).expand(-1, self.pred_len, -1)

    @staticmethod
    def build_gate_input_g0(pred_h: torch.Tensor, pred_c: torch.Tensor) -> torch.Tensor:
        return torch.cat((pred_h, pred_c, pred_c - pred_h), dim=-1)

    def _build_gate_input(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h_raw: torch.Tensor,
        gate_feat_c_raw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gate_feat_h = self._expand_gate_summary(self._pool_branch_feature(gate_feat_h_raw))
        gate_feat_c = self._expand_gate_summary(self._pool_branch_feature(gate_feat_c_raw))
        if self.gate_input_mode in {"g1b-topclip", "g1b-topclip-lite"}:
            gate_feat_h = self.gate_hidden_scale * gate_feat_h
            gate_feat_c = self.gate_hidden_scale * gate_feat_c
        gate_input = torch.cat(
            (gate_feat_h, gate_feat_c, self.build_gate_input_g0(pred_h, pred_c)),
            dim=-1,
        )
        return {
            "gate_input": gate_input,
            "gate_feat_h": gate_feat_h,
            "gate_feat_c": gate_feat_c,
        }

    @staticmethod
    def _constant_gate(reference: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full_like(reference, fill_value=value)

    def forward(
        self,
        pred_h: torch.Tensor,
        pred_c: torch.Tensor,
        gate_feat_h_raw: torch.Tensor,
        gate_feat_c_raw: torch.Tensor,
        has_context: bool,
    ) -> Dict[str, Any]:
        if not has_context or self.stic_mode == "history_only":
            gate = self._constant_gate(pred_h, 0.0)
            return {
                "pred": pred_h,
                "gate": gate,
                "gate_trainable": False,
                "aux_loss_enabled": False,
                "mode": "history_only" if self.stic_mode == "history_only" else "no_context",
            }

        if self.stic_mode in {"always_on", "no_gate"}:
            gate = self._constant_gate(pred_h, 1.0)
            return {
                "pred": pred_c,
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

        extra_outputs: Dict[str, torch.Tensor] = {}
        if self.gate_input_mode == "g0":
            gate_input = self.build_gate_input_g0(pred_h, pred_c)
        else:
            extra_outputs = self._build_gate_input(
                pred_h=pred_h,
                pred_c=pred_c,
                gate_feat_h_raw=gate_feat_h_raw,
                gate_feat_c_raw=gate_feat_c_raw,
            )
            gate_input = extra_outputs["gate_input"]

        gate = self.gate_network(gate_input)
        pred = pred_h + gate * (pred_c - pred_h)
        outputs: Dict[str, Any] = {
            "pred": pred,
            "gate": gate,
            "gate_trainable": True,
            "aux_loss_enabled": True,
            "mode": "dynamic",
            "gate_input": gate_input,
        }
        outputs.update(extra_outputs)
        return outputs
