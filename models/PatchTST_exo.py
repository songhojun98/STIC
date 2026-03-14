from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.PatchTST import Model as PatchTSTBackbone
from models._exo_stic_common import (
    SelectiveTrustGate,
    TargetSequenceMixer,
    clone_configs,
    move_target_last,
    resolve_target_index,
    select_target_channel,
)


class _SafePatchTSTBackbone(PatchTSTBackbone):
    """PatchTST variant that avoids in-place normalization on grad-carrying inputs."""

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        x_mark_dec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


class _PatchTSTContextBranch(nn.Module):
    """Inject exogenous information into a target-specific sequence before PatchTST."""

    def __init__(self, configs: Any, input_channels: int) -> None:
        super().__init__()
        self.mixer = TargetSequenceMixer(
            input_channels=input_channels,
            target_index=max(input_channels - 1, 0),
            mixer_type=getattr(configs, "stic_context_mixer_type", "linear"),
            mixer_hidden_dim=getattr(configs, "stic_context_mixer_hidden_dim", 0),
            residual_scale=getattr(configs, "stic_context_residual_scale", 0.5),
        )
        branch_configs = clone_configs(
            configs,
            enc_in=1,
            dec_in=1,
            c_out=1,
            features="S",
        )
        self.temporal_head = _SafePatchTSTBackbone(branch_configs)

    def encode(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        x_mark_dec: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mixed = self.mixer.encode(x_enc)
        # PatchTST normalizes its input in-place, so the gate feature path must
        # keep an untouched copy of the mixed target sequence.
        temporal_input = mixed["mixed_sequence"].clone()
        pred = self.temporal_head(
            temporal_input,
            x_mark_enc,
            x_dec[..., -1:].clone(),
            x_mark_dec,
        )
        return {
            "pred": pred,
            "mixed_sequence": mixed["mixed_sequence"],
            "mixed_residual": mixed["mixed_residual"],
        }

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        x_mark_dec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)["pred"]


class Model(nn.Module):
    """STIC-style exogenous wrapper around PatchTST."""

    def __init__(self, configs: Any) -> None:
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                "PatchTST_exo currently supports only forecasting tasks."
            )

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.enc_in = int(configs.enc_in)
        self.target_index = resolve_target_index(configs)

        history_configs = clone_configs(
            configs,
            enc_in=1,
            dec_in=1,
            c_out=1,
            features="S",
        )
        self.history_branch = _SafePatchTSTBackbone(history_configs)
        self.context_branch = _PatchTSTContextBranch(configs, input_channels=self.enc_in)
        self.gate = SelectiveTrustGate(configs, pred_len=self.pred_len)

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        x_mark_dec: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        target_enc = select_target_channel(x_enc, self.target_index)
        target_dec = select_target_channel(x_dec, self.target_index)
        pred_h = self.history_branch(
            target_enc.clone(),
            x_mark_enc,
            target_dec.clone(),
            x_mark_dec,
        )

        has_context = x_enc.size(-1) > 1
        if has_context:
            context_enc = move_target_last(x_enc, self.target_index).clone()
            context_dec = move_target_last(x_dec, self.target_index).clone()
            context_outputs = self.context_branch.encode(
                context_enc, x_mark_enc, context_dec, x_mark_dec
            )
            pred_c = context_outputs["pred"]
            gate_feat_c_raw = context_outputs["mixed_sequence"].clone()
        else:
            pred_c = pred_h
            gate_feat_c_raw = target_enc.clone()

        outputs = self.gate(
            pred_h=pred_h,
            pred_c=pred_c,
            gate_feat_h_raw=target_enc.clone(),
            gate_feat_c_raw=gate_feat_c_raw,
            has_context=has_context,
        )
        outputs["pred_h"] = pred_h
        outputs["pred_c"] = pred_c
        outputs["gate_input_mode"] = self.gate.gate_input_mode
        outputs["gate_feat_h_raw"] = target_enc.clone()
        outputs["gate_feat_c_raw"] = gate_feat_c_raw
        return outputs

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        x_mark_dec: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del mask
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
