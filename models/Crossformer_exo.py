from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.Crossformer import Model as CrossformerBackbone
from models._exo_stic_common import (
    SelectiveTrustGate,
    TargetSequenceMixer,
    clone_configs,
    move_target_last,
    resolve_target_index,
    select_target_channel,
)


class Model(nn.Module):
    """STIC-style exogenous wrapper around Crossformer."""

    def __init__(self, configs: Any) -> None:
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                "Crossformer_exo currently supports only forecasting tasks."
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
        context_configs = clone_configs(
            configs,
            enc_in=self.enc_in,
            dec_in=self.enc_in,
            c_out=self.enc_in,
            features="M",
        )

        self.history_branch = CrossformerBackbone(history_configs)
        self.context_branch = CrossformerBackbone(context_configs)
        self.context_hint_mixer = TargetSequenceMixer(
            input_channels=self.enc_in,
            target_index=max(self.enc_in - 1, 0),
            mixer_type=getattr(configs, "stic_context_mixer_type", "linear"),
            mixer_hidden_dim=getattr(configs, "stic_context_mixer_hidden_dim", 0),
            residual_scale=getattr(configs, "stic_context_residual_scale", 0.5),
        )
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
        pred_h = self.history_branch(target_enc, x_mark_enc, target_dec, x_mark_dec)

        has_context = x_enc.size(-1) > 1
        if has_context:
            context_enc = move_target_last(x_enc, self.target_index)
            context_dec = move_target_last(x_dec, self.target_index)
            pred_c_full = self.context_branch(
                context_enc, x_mark_enc, context_dec, x_mark_dec
            )
            pred_c = pred_c_full[..., -1:]
            gate_feat_c_raw = self.context_hint_mixer(context_enc)
        else:
            pred_c = pred_h
            gate_feat_c_raw = target_enc

        outputs = self.gate(
            pred_h=pred_h,
            pred_c=pred_c,
            gate_feat_h_raw=target_enc,
            gate_feat_c_raw=gate_feat_c_raw,
            has_context=has_context,
        )
        outputs["pred_h"] = pred_h
        outputs["pred_c"] = pred_c
        outputs["gate_input_mode"] = self.gate.gate_input_mode
        outputs["gate_feat_h_raw"] = target_enc
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
