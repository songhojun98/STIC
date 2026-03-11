from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_TIMESTAMP_RE = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"


@dataclass
class STICCiKOutput:
    """Container for the thin CiK STIC prototype outputs."""

    pred: np.ndarray
    pred_h: np.ndarray
    pred_c: np.ndarray
    gate: np.ndarray
    rule_name: str
    confidence: float
    notes: Tuple[str, ...]
    seasonal_period: int
    noise_scale: float
    applied_positions: Tuple[int, ...]

    def sample_paths(self, n_samples: int, seed: int) -> np.ndarray:
        """Return probabilistic sample paths around the gated point forecast."""

        return self.sample_paths_for_prediction(
            prediction=self.pred,
            n_samples=n_samples,
            seed=seed,
        )

    def sample_paths_for_branch(self, branch_name: str, n_samples: int, seed: int) -> np.ndarray:
        """Return sample paths around a selected branch forecast."""

        branch_predictions = {
            "pred": self.pred,
            "pred_final": self.pred,
            "pred_h": self.pred_h,
            "history_only": self.pred_h,
            "pred_c": self.pred_c,
            "context_aware": self.pred_c,
        }
        if branch_name not in branch_predictions:
            available = ", ".join(sorted(branch_predictions))
            raise KeyError(
                f"Unknown branch '{branch_name}'. Available branches: {available}"
            )
        return self.sample_paths_for_prediction(
            prediction=branch_predictions[branch_name],
            n_samples=n_samples,
            seed=seed,
        )

    def sample_paths_for_prediction(
        self,
        prediction: Sequence[float],
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Return probabilistic sample paths around an arbitrary point forecast."""

        center_values = np.asarray(prediction, dtype=np.float32)
        horizon = int(center_values.shape[0])
        center = center_values.reshape(1, horizon, 1)
        if n_samples <= 1 or self.noise_scale <= 0:
            return np.repeat(center, repeats=max(1, n_samples), axis=0)

        rng = np.random.default_rng(seed)
        samples = center + rng.normal(
            loc=0.0,
            scale=self.noise_scale,
            size=(n_samples, horizon, 1),
        )
        return samples.astype(np.float32)


@dataclass
class ParsedEffect:
    """Task-specific text effect applied only to the context-aware branch."""

    rule_name: str
    confidence: float
    notes: Tuple[str, ...]
    mask: np.ndarray
    apply_fn: Callable[[np.ndarray], np.ndarray]
    gate_hint: Optional[np.ndarray] = None

    def apply(self, prediction: np.ndarray) -> np.ndarray:
        """Apply the parsed effect to a forecast copy."""

        return self.apply_fn(prediction.copy())

    @property
    def active_positions(self) -> Tuple[int, ...]:
        """Return horizon indices where the effect is active."""

        return tuple(int(index) for index in np.flatnonzero(self.mask))


class STICCiKPrototype:
    """Thin STIC prototype for CiK text-context tasks.

    The prototype keeps the STIC semantics intact:
    1. `pred_h`: history-only forecast from past target values
    2. `pred_c`: text-conditioned context-aware forecast
    3. `gate`: simple g0-style trust gate derived from `[pred_h, pred_c, pred_c-pred_h]`
    4. `pred = pred_h + gate * (pred_c - pred_h)`
    """

    def __init__(
        self,
        gate_floor: float = 0.05,
        gate_context_scale: float = 1.0,
        gate_delta_scale: float = 1.5,
        gate_bias: float = -2.4,
        noise_scale_factor: float = 0.15,
    ) -> None:
        self.gate_floor = float(np.clip(gate_floor, 0.0, 1.0))
        self.gate_context_scale = float(max(gate_context_scale, 0.0))
        self.gate_delta_scale = float(max(gate_delta_scale, 0.0))
        self.gate_bias = float(gate_bias)
        self.noise_scale_factor = float(max(noise_scale_factor, 0.0))

    def forward(
        self,
        history_frame: pd.DataFrame,
        future_frame: pd.DataFrame,
        context_text: str,
        task_name: str = "",
    ) -> STICCiKOutput:
        """Run history-only, context-aware, and gated forecasts for one sample."""

        history = history_frame.iloc[:, -1].astype(float).to_numpy()
        future_index = future_frame.index
        horizon = len(future_frame)
        seasonal_period = infer_seasonal_period(history_frame.index, len(history))
        pred_h = history_only_forecast(
            history=history,
            horizon=horizon,
            seasonal_period=seasonal_period,
        )
        pred_c = pred_h.copy()

        effects = []
        for parser in (
            parse_electricity_effect,
            parse_sensor_maintenance_effect,
        ):
            effect = parser(context_text=context_text, future_index=future_index)
            if effect is not None:
                effects.append(effect)

        constraint_effect = parse_constraint_effect(context_text=context_text, base_prediction=pred_h)
        if constraint_effect is not None:
            effects.append(constraint_effect)

        notes = []
        rule_names = []
        confidence = 0.0
        applied_positions = set()
        gate_hint = np.full(horizon, self.gate_floor, dtype=np.float32)

        for effect in effects:
            pred_c = effect.apply(pred_c)
            rule_names.append(effect.rule_name)
            confidence = max(confidence, effect.confidence)
            notes.extend(effect.notes)
            applied_positions.update(effect.active_positions)
            gate_hint = np.maximum(gate_hint, self._effect_gate_hint(effect, horizon))

        gate = compute_prediction_gate(
            pred_h=pred_h,
            pred_c=pred_c,
            gate_hint=gate_hint,
            gate_floor=self.gate_floor,
            gate_delta_scale=self.gate_delta_scale,
            gate_bias=self.gate_bias,
        )
        pred = pred_h + gate * (pred_c - pred_h)
        rule_name = "+".join(rule_names) if rule_names else "history_only"
        noise_scale = estimate_noise_scale(history, seasonal_period) * self.noise_scale_factor

        if effects:
            notes.append(
                "active_positions="
                + describe_positions(tuple(sorted(applied_positions)), horizon=horizon)
            )

        return STICCiKOutput(
            pred=pred.astype(np.float32),
            pred_h=pred_h.astype(np.float32),
            pred_c=pred_c.astype(np.float32),
            gate=gate.astype(np.float32),
            rule_name=rule_name,
            confidence=float(confidence),
            notes=tuple(notes),
            seasonal_period=int(seasonal_period),
            noise_scale=float(noise_scale),
            applied_positions=tuple(sorted(applied_positions)),
        )

    def _effect_gate_hint(self, effect: ParsedEffect, horizon: int) -> np.ndarray:
        """Convert an effect into a coarse trust prior before prediction gating."""

        if effect.gate_hint is not None:
            return np.asarray(effect.gate_hint, dtype=np.float32)

        hint = np.full(horizon, self.gate_floor, dtype=np.float32)
        if effect.mask.any():
            active_value = float(
                np.clip(effect.confidence * self.gate_context_scale, self.gate_floor, 1.0)
            )
            hint[effect.mask] = active_value
        return hint


def infer_seasonal_period(index: pd.Index, history_length: int) -> int:
    """Infer a simple seasonal period from the timestamp frequency."""

    if history_length < 2:
        return 1

    step = infer_step(index)
    if step is None:
        return 1
    if step <= pd.Timedelta(minutes=15):
        return 96 if history_length >= 96 else 1
    if step <= pd.Timedelta(minutes=30):
        return 48 if history_length >= 48 else 1
    if step <= pd.Timedelta(hours=1):
        return 24 if history_length >= 24 else 1
    if step <= pd.Timedelta(days=1):
        return 7 if history_length >= 7 else 1
    if step <= pd.Timedelta(weeks=1):
        return 4 if history_length >= 4 else 1
    return 1


def infer_step(index: pd.Index) -> Optional[pd.Timedelta]:
    """Infer the time delta between consecutive timestamps."""

    if len(index) < 2:
        return None
    if isinstance(index, pd.PeriodIndex):
        values = index.to_timestamp()
    else:
        values = pd.to_datetime(index, errors="coerce")
    if len(values) < 2 or pd.isna(values[0]) or pd.isna(values[1]):
        return None
    return values[1] - values[0]


def history_only_forecast(
    history: np.ndarray,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """Simple history-only baseline: seasonal replay plus a clipped tail trend."""

    history = np.asarray(history, dtype=np.float32)
    if history.size == 0:
        return np.zeros(horizon, dtype=np.float32)

    if seasonal_period > 1 and history.size >= seasonal_period:
        template = np.resize(history[-seasonal_period:], horizon).astype(np.float32)
        slope = robust_tail_slope(history, window=max(4, min(seasonal_period, history.size)))
        trend = 0.25 * slope * np.arange(1, horizon + 1, dtype=np.float32)
        return template + trend

    slope = robust_tail_slope(history, window=min(12, history.size))
    return history[-1] + slope * np.arange(1, horizon + 1, dtype=np.float32)


def robust_tail_slope(history: np.ndarray, window: int) -> float:
    """Estimate a robust linear trend from the tail of the history."""

    window = max(2, min(window, history.size))
    tail = history[-window:]
    diffs = np.diff(tail)
    if diffs.size == 0:
        return 0.0
    raw_slope = float(np.median(diffs))
    scale = max(float(np.std(tail)), 1e-6)
    return float(np.clip(raw_slope, -0.5 * scale, 0.5 * scale))


def estimate_noise_scale(history: np.ndarray, seasonal_period: int) -> float:
    """Estimate residual noise for simple probabilistic sampling."""

    history = np.asarray(history, dtype=np.float32)
    if history.size < 3:
        return 0.0
    if seasonal_period > 1 and history.size >= 2 * seasonal_period:
        residual = history[-seasonal_period:] - history[-2 * seasonal_period : -seasonal_period]
    else:
        residual = np.diff(history[-min(history.size, 16) :])
    return float(np.std(residual))


def compute_prediction_gate(
    pred_h: Sequence[float],
    pred_c: Sequence[float],
    gate_hint: Sequence[float],
    gate_floor: float,
    gate_delta_scale: float,
    gate_bias: float,
) -> np.ndarray:
    """Compute a simple g0-style prediction-level gate from `[pred_h, pred_c, pred_c-pred_h]`."""

    pred_h_arr = np.asarray(pred_h, dtype=np.float32)
    pred_c_arr = np.asarray(pred_c, dtype=np.float32)
    gate_hint_arr = np.asarray(gate_hint, dtype=np.float32)

    scale = max(
        float(np.mean(np.abs(pred_h_arr))),
        float(np.std(pred_h_arr)),
        float(np.std(pred_c_arr)),
        1.0,
    )
    q_h = pred_h_arr / scale
    q_c = pred_c_arr / scale
    q_delta = np.clip(np.abs(pred_c_arr - pred_h_arr) / scale, 0.0, 1.0)
    hint_strength = np.clip((gate_hint_arr - gate_floor) / max(1.0 - gate_floor, 1e-6), 0.0, 1.0)
    logits = gate_bias + (0.15 * q_h) + (0.15 * q_c) + (gate_delta_scale * q_delta) + (1.75 * hint_strength)
    gate = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(np.maximum(gate, gate_floor), gate_floor, 1.0).astype(np.float32)


def parse_electricity_effect(
    context_text: str,
    future_index: pd.Index,
) -> Optional[ParsedEffect]:
    """Parse heat-wave electricity increase text into a horizon-local multiplier or shift."""

    text = context_text or ""
    lowered = text.lower()
    if "heatwave" not in lowered and "heat wave" not in lowered and "electricity" not in lowered:
        return None

    start_match = re.search(
        rf"(?:from|began on)\s+(?P<start>{_TIMESTAMP_RE})",
        text,
        flags=re.IGNORECASE,
    )
    duration_match = re.search(
        r"(?:for|lasted for(?: approximately)?)\s+(?P<duration>\d+)\s+hour",
        text,
        flags=re.IGNORECASE,
    )
    multiplier_match = re.search(
        r"(?P<multiplier>\d+(?:\.\d+)?)\s+times",
        text,
        flags=re.IGNORECASE,
    )
    percent_match = re.search(
        r"increase(?:d)?(?: by)?\s+(?P<percent>\d+(?:\.\d+)?)\s*%",
        text,
        flags=re.IGNORECASE,
    )
    additive_match = re.search(
        r"increase(?:d)?(?: by)?\s+(?P<delta>\d+(?:\.\d+)?)\s+(?:kw|kilowatt|units?)",
        lowered,
        flags=re.IGNORECASE,
    )
    if start_match is None or duration_match is None:
        return None

    start = pd.Timestamp(start_match.group("start"))
    duration = int(duration_match.group("duration"))
    mask = build_position_mask(
        future_index=future_index,
        start=start,
        duration=duration,
        closed="left",
    )
    if not mask.any():
        return None

    multiplier = float(multiplier_match.group("multiplier")) if multiplier_match else None
    additive_delta = float(additive_match.group("delta")) if additive_match else None
    if multiplier is None and percent_match is not None:
        multiplier = 1.0 + (float(percent_match.group("percent")) / 100.0)

    confidence = 0.95
    if "did not affect" in lowered or "not expected to happen" in lowered:
        confidence -= 0.15
    if "neighbouring" in lowered or "nearby city" in lowered or "distractor" in lowered:
        confidence -= 0.05
    confidence = float(np.clip(confidence, 0.5, 0.98))

    def apply_fn(prediction: np.ndarray) -> np.ndarray:
        if multiplier is not None:
            prediction[mask] = prediction[mask] * multiplier
        elif additive_delta is not None:
            prediction[mask] = prediction[mask] + additive_delta
        return prediction

    notes = [f"start={start}", f"duration={duration}"]
    if multiplier is not None:
        notes.append(f"multiplier={multiplier:.4f}")
    if additive_delta is not None:
        notes.append(f"additive_delta={additive_delta:.4f}")

    return ParsedEffect(
        rule_name="electricity_event",
        confidence=confidence,
        notes=tuple(notes),
        mask=mask,
        apply_fn=apply_fn,
    )


def parse_sensor_maintenance_effect(
    context_text: str,
    future_index: pd.Index,
) -> Optional[ParsedEffect]:
    """Parse explicit zero-reading maintenance events into a context adjustment."""

    text = context_text or ""
    lowered = text.lower()
    if "maintenance" not in lowered or "offline" not in lowered:
        return None

    pattern = re.compile(
        rf"between\s+(?P<start>{_TIMESTAMP_RE})\s+and\s+(?P<end>{_TIMESTAMP_RE}).*?zero readings",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        return None

    start = pd.Timestamp(match.group("start"))
    end = pd.Timestamp(match.group("end"))
    mask = build_time_mask(
        future_index=future_index,
        start=start,
        end=end,
        closed="left",
    )
    if not mask.any():
        return None

    def apply_fn(prediction: np.ndarray) -> np.ndarray:
        prediction[mask] = 0.0
        return prediction

    return ParsedEffect(
        rule_name="sensor_offline_zero",
        confidence=0.95,
        notes=(f"start={start}", f"end={end}"),
        mask=mask,
        apply_fn=apply_fn,
    )


def parse_constraint_effect(
    context_text: str,
    base_prediction: np.ndarray,
) -> Optional[ParsedEffect]:
    """Parse simple lower/upper textual bounds into a clipping effect."""

    text = context_text or ""
    lowered = text.lower()
    if "bounded below" not in lowered and "bounded above" not in lowered:
        return None

    lower_match = re.search(r"bounded below by\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    upper_match = re.search(r"bounded above by\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    lower_bound = float(lower_match.group(1)) if lower_match else None
    upper_bound = float(upper_match.group(1)) if upper_match else None

    violation_mask = np.zeros(base_prediction.shape[0], dtype=bool)
    gate_hint = np.full(base_prediction.shape[0], 0.05, dtype=np.float32)
    notes = []
    if lower_bound is not None:
        lower_violation = np.maximum(lower_bound - base_prediction, 0.0)
        lower_mask = lower_violation > 0
        violation_mask = np.logical_or(violation_mask, lower_mask)
        if lower_mask.any():
            gate_hint[lower_mask] = np.maximum(
                gate_hint[lower_mask],
                np.clip(0.30 + lower_violation[lower_mask], 0.30, 1.0),
            )
        notes.append(f"lower={lower_bound:.4f}")
    if upper_bound is not None:
        upper_violation = np.maximum(base_prediction - upper_bound, 0.0)
        upper_mask = upper_violation > 0
        violation_mask = np.logical_or(violation_mask, upper_mask)
        if upper_mask.any():
            gate_hint[upper_mask] = np.maximum(
                gate_hint[upper_mask],
                np.clip(0.30 + upper_violation[upper_mask], 0.30, 1.0),
            )
        notes.append(f"upper={upper_bound:.4f}")

    def apply_fn(prediction: np.ndarray) -> np.ndarray:
        adjusted = prediction.copy()
        if lower_bound is not None:
            adjusted = np.maximum(adjusted, lower_bound)
        if upper_bound is not None:
            adjusted = np.minimum(adjusted, upper_bound)
        return adjusted

    if not notes:
        return None

    return ParsedEffect(
        rule_name="forecast_constraints",
        confidence=0.85,
        notes=tuple(notes),
        mask=violation_mask,
        apply_fn=apply_fn,
        gate_hint=gate_hint,
    )


def build_position_mask(
    future_index: pd.Index,
    start: pd.Timestamp,
    duration: int,
    closed: str = "left",
) -> np.ndarray:
    """Build a boolean mask from a start timestamp and a duration."""

    step = infer_step(future_index)
    if step is None:
        values = pd.to_datetime(future_index, errors="coerce")
        if len(values) == 0:
            return np.zeros(0, dtype=bool)
        match_positions = np.where(values == start)[0]
        if match_positions.size == 0:
            nearest = int(np.argmin(np.abs(values - start)))
            match_positions = np.array([nearest], dtype=int)
        start_idx = int(match_positions[0])
        end_idx = min(start_idx + max(duration, 0), len(values))
        mask = np.zeros(len(values), dtype=bool)
        mask[start_idx:end_idx] = True
        return mask

    end = start + (duration * step)
    return build_time_mask(future_index=future_index, start=start, end=end, closed=closed)


def build_time_mask(
    future_index: pd.Index,
    start: pd.Timestamp,
    end: pd.Timestamp,
    closed: str = "left",
) -> np.ndarray:
    """Build a boolean time mask over a forecast horizon index."""

    if isinstance(future_index, pd.PeriodIndex):
        values = future_index.to_timestamp()
    else:
        values = pd.to_datetime(future_index, errors="coerce")
    if closed == "left":
        return np.logical_and(values >= start, values < end)
    return np.logical_and(values >= start, values <= end)


def describe_positions(indices: Sequence[int], horizon: int) -> str:
    """Format active horizon indices for human-readable debug output."""

    if not indices:
        return "none"
    preview = list(indices[:8])
    suffix = "" if len(indices) <= 8 else f"...(+{len(indices) - 8})"
    return f"{preview}{suffix} / horizon={horizon}"


def compute_basic_metrics(
    truth: Sequence[float],
    pred: Sequence[float],
    pred_h: Sequence[float],
    pred_c: Sequence[float],
    gate: Sequence[float],
    region_of_interest: Sequence[int],
) -> Dict[str, float]:
    """Compute basic scalar metrics for history-only, context-aware, and gated outputs."""

    truth_arr = np.asarray(truth, dtype=np.float32)
    pred_arr = np.asarray(pred, dtype=np.float32)
    pred_h_arr = np.asarray(pred_h, dtype=np.float32)
    pred_c_arr = np.asarray(pred_c, dtype=np.float32)
    gate_arr = np.asarray(gate, dtype=np.float32)

    roi_mask = np.zeros(truth_arr.shape[0], dtype=bool)
    if region_of_interest:
        valid_indices = [idx for idx in region_of_interest if 0 <= int(idx) < truth_arr.shape[0]]
        if valid_indices:
            roi_mask[np.asarray(valid_indices, dtype=int)] = True
    non_roi_mask = ~roi_mask if roi_mask.any() else np.zeros_like(roi_mask)

    metrics = {
        "mse": float(np.mean((pred_arr - truth_arr) ** 2)),
        "mae": float(np.mean(np.abs(pred_arr - truth_arr))),
        "rmse": float(math.sqrt(np.mean((pred_arr - truth_arr) ** 2))),
        "mse_h": float(np.mean((pred_h_arr - truth_arr) ** 2)),
        "mae_h": float(np.mean(np.abs(pred_h_arr - truth_arr))),
        "mse_c": float(np.mean((pred_c_arr - truth_arr) ** 2)),
        "mae_c": float(np.mean(np.abs(pred_c_arr - truth_arr))),
        "gate_mean": float(np.mean(gate_arr)),
        "gate_std": float(np.std(gate_arr)),
        "pred_h_mean": float(np.mean(pred_h_arr)),
        "pred_c_mean": float(np.mean(pred_c_arr)),
        "pred_final_mean": float(np.mean(pred_arr)),
        "pred_c_delta_mean_abs": float(np.mean(np.abs(pred_c_arr - pred_h_arr))),
        "pred_final_delta_mean_abs": float(np.mean(np.abs(pred_arr - pred_h_arr))),
    }

    if roi_mask.any():
        metrics["roi_mse"] = float(np.mean((pred_arr[roi_mask] - truth_arr[roi_mask]) ** 2))
        metrics["roi_mae"] = float(np.mean(np.abs(pred_arr[roi_mask] - truth_arr[roi_mask])))
        metrics["roi_mse_h"] = float(np.mean((pred_h_arr[roi_mask] - truth_arr[roi_mask]) ** 2))
        metrics["roi_mae_h"] = float(np.mean(np.abs(pred_h_arr[roi_mask] - truth_arr[roi_mask])))
        metrics["roi_mse_c"] = float(np.mean((pred_c_arr[roi_mask] - truth_arr[roi_mask]) ** 2))
        metrics["roi_mae_c"] = float(np.mean(np.abs(pred_c_arr[roi_mask] - truth_arr[roi_mask])))
        metrics["gate_roi_mean"] = float(np.mean(gate_arr[roi_mask]))
    else:
        metrics["roi_mse"] = float("nan")
        metrics["roi_mae"] = float("nan")
        metrics["roi_mse_h"] = float("nan")
        metrics["roi_mae_h"] = float("nan")
        metrics["roi_mse_c"] = float("nan")
        metrics["roi_mae_c"] = float("nan")
        metrics["gate_roi_mean"] = float("nan")

    if non_roi_mask.any():
        metrics["non_roi_mse"] = float(np.mean((pred_arr[non_roi_mask] - truth_arr[non_roi_mask]) ** 2))
        metrics["non_roi_mae"] = float(np.mean(np.abs(pred_arr[non_roi_mask] - truth_arr[non_roi_mask])))
        metrics["gate_non_roi_mean"] = float(np.mean(gate_arr[non_roi_mask]))
    else:
        metrics["non_roi_mse"] = float("nan")
        metrics["non_roi_mae"] = float("nan")
        metrics["gate_non_roi_mean"] = float("nan")

    metrics["mse_gain_vs_h"] = metrics["mse_h"] - metrics["mse"]
    metrics["mae_gain_vs_h"] = metrics["mae_h"] - metrics["mae"]
    metrics["mse_gain_vs_c"] = metrics["mse_c"] - metrics["mse"]
    metrics["mae_gain_vs_c"] = metrics["mae_c"] - metrics["mae"]
    metrics["roi_gate_alignment"] = metrics["gate_roi_mean"] - metrics["gate_non_roi_mean"]
    return metrics


__all__ = [
    "STICCiKOutput",
    "STICCiKPrototype",
    "compute_basic_metrics",
    "compute_prediction_gate",
    "describe_positions",
    "history_only_forecast",
    "infer_seasonal_period",
    "parse_constraint_effect",
    "parse_electricity_effect",
    "parse_sensor_maintenance_effect",
]
