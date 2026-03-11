from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

CIK_DATASET_ID = "ServiceNow/context-is-key"
DEFAULT_CIK_SPLIT = "test"
DEFAULT_CONTEXT_FIELDS = ("scenario", "background", "constraints")
DEFAULT_CIK_ROI_WEIGHT = 0.5
STIC_CIK_TASK_PRESETS: Dict[str, Tuple[str, ...]] = {
    "stic_minimal": (
        "ElectricityIncreaseInPredictionTask",
        "OraclePredUnivariateConstraintsTask",
    ),
    "stic_extension": (
        "ElectricityIncreaseInPredictionWithDistractorText",
        "ShortNewsElectricityIncreaseInPredictionTask",
        "MediumNewsElectricityIncreaseInPredictionTask",
        "LongNewsElectricityIncreaseInPredictionTask",
    ),
    "stic_next": (
        "SensorMaintenanceInPredictionTask",
    ),
    "stic_all_recommended": (
        "ElectricityIncreaseInPredictionTask",
        "OraclePredUnivariateConstraintsTask",
        "ElectricityIncreaseInPredictionWithDistractorText",
        "ShortNewsElectricityIncreaseInPredictionTask",
        "MediumNewsElectricityIncreaseInPredictionTask",
        "LongNewsElectricityIncreaseInPredictionTask",
        "SensorMaintenanceInPredictionTask",
    ),
}

_OFFICIAL_METRIC_PACKAGE = "stic_cik_metrics_compat"
_OFFICIAL_THRESHOLD_WEIGHTED_CRPS = None
_OFFICIAL_CONSTRAINT_CLASSES = None


def resolve_cik_cache_dir(cache_dir: Optional[str] = None) -> Optional[str]:
    if cache_dir:
        return cache_dir

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return hf_home

    cik_data_store = os.getenv("CIK_DATA_STORE")
    if cik_data_store:
        return os.path.join(cik_data_store, "hf_cache")

    return None


def load_cik_rows(
    split: str = DEFAULT_CIK_SPLIT,
    task_name: Optional[str] = None,
    task_names: Optional[Sequence[str]] = None,
    preset_name: Optional[str] = None,
    limit: Optional[int] = None,
    limit_per_task: bool = False,
    cache_dir: Optional[str] = None,
) -> HFDataset:
    resolved_cache_dir = resolve_cik_cache_dir(cache_dir)
    dataset = load_dataset(CIK_DATASET_ID, split=split, cache_dir=resolved_cache_dir)

    requested_names = resolve_cik_task_names(
        task_name=task_name,
        task_names=task_names,
        preset_name=preset_name,
    )
    if requested_names:
        allowed_names = set(requested_names)
        indices = [
            row_index
            for row_index, current_name in enumerate(dataset["name"])
            if current_name in allowed_names
        ]
        dataset = dataset.select(indices)

    if limit is not None:
        resolved_limit = max(0, int(limit))
        if limit_per_task and requested_names:
            kept_indices: List[int] = []
            per_task_counts: Counter[str] = Counter()
            for row_index, current_name in enumerate(dataset["name"]):
                if per_task_counts[current_name] >= resolved_limit:
                    continue
                kept_indices.append(row_index)
                per_task_counts[current_name] += 1
            dataset = dataset.select(kept_indices)
        else:
            dataset = dataset.select(range(min(resolved_limit, len(dataset))))

    return dataset


def list_cik_task_names(
    split: str = DEFAULT_CIK_SPLIT,
    cache_dir: Optional[str] = None,
) -> List[str]:
    dataset = load_cik_rows(split=split, cache_dir=cache_dir)
    return sorted(set(dataset["name"]))


def list_cik_task_presets() -> Dict[str, Tuple[str, ...]]:
    return dict(STIC_CIK_TASK_PRESETS)


def get_cik_task_preset(preset_name: str) -> Tuple[str, ...]:
    if preset_name not in STIC_CIK_TASK_PRESETS:
        available = ", ".join(sorted(STIC_CIK_TASK_PRESETS))
        raise KeyError(
            f"Unknown CiK preset '{preset_name}'. Available presets: {available}"
        )
    return STIC_CIK_TASK_PRESETS[preset_name]


def resolve_cik_task_names(
    task_name: Optional[str] = None,
    task_names: Optional[Sequence[str]] = None,
    preset_name: Optional[str] = None,
) -> Tuple[str, ...]:
    normalized = list(_normalize_task_names(task_name=task_name, task_names=task_names))
    if preset_name:
        normalized.extend(get_cik_task_preset(preset_name))
    return tuple(dict.fromkeys(normalized))


def parse_cik_timeframe(payload: Any) -> pd.DataFrame:
    if isinstance(payload, str):
        parsed_payload = json.loads(payload)
    elif isinstance(payload, Mapping):
        parsed_payload = payload
    else:
        raise TypeError(
            "Unsupported CiK timeframe payload. Expected a JSON string or mapping."
        )

    frame = pd.DataFrame(parsed_payload)
    frame.columns = [str(column_name) for column_name in frame.columns]

    parsed_index = pd.to_datetime(frame.index, errors="coerce")
    if len(frame.index) and not parsed_index.isna().any():
        frame.index = parsed_index
        frame = frame.sort_index()
    else:
        frame = frame.sort_index()

    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("CiK timeframe contains non-numeric values after parsing.")

    return frame


def merge_cik_context(
    row: Mapping[str, Any],
    context_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
    include_field_names: bool = True,
) -> Tuple[str, Dict[str, str]]:
    merged_parts: List[str] = []
    context_map: Dict[str, str] = {}

    for field_name in context_fields:
        raw_value = row.get(field_name, "")
        if not isinstance(raw_value, str):
            continue
        value = raw_value.strip()
        if not value:
            continue
        context_map[field_name] = value
        if include_field_names:
            label = field_name.replace("_", " ").strip()
            merged_parts.append(f"{label}: {value}")
        else:
            merged_parts.append(value)

    return "\n\n".join(merged_parts), context_map


def select_cik_target_column(
    frame: pd.DataFrame,
    target_column: Optional[str | int] = None,
) -> str:
    if frame.empty:
        raise ValueError("Cannot select a target column from an empty CiK frame.")

    if target_column is None:
        return str(frame.columns[-1])

    if isinstance(target_column, int):
        return str(frame.columns[target_column])

    if target_column not in frame.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the CiK frame.")

    return str(target_column)


def frame_to_target_tensor(
    frame: pd.DataFrame,
    target_column: Optional[str | int] = None,
) -> Tuple[torch.Tensor, str]:
    resolved_target_column = select_cik_target_column(frame, target_column=target_column)
    series = frame[resolved_target_column].astype(float).to_numpy()
    tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(-1)
    return tensor, resolved_target_column


@dataclass
class CIKSample:
    row_index: int
    task_name: str
    target_column: str
    history_frame: pd.DataFrame
    future_frame: pd.DataFrame
    history: torch.Tensor
    future: torch.Tensor
    context_text: str
    context_fields: Dict[str, str]
    context_sources: Tuple[str, ...]
    metric_scaling: float
    region_of_interest: Tuple[int, ...]
    metadata: Dict[str, Any]

    def to_batch_item(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "task_name": self.task_name,
            "target_column": self.target_column,
            "history_frame": self.history_frame,
            "future_frame": self.future_frame,
            "history": self.history,
            "future": self.future,
            "history_timestamps": tuple(str(index_value) for index_value in self.history_frame.index),
            "future_timestamps": tuple(str(index_value) for index_value in self.future_frame.index),
            "context_text": self.context_text,
            "context_fields": self.context_fields,
            "context_sources": self.context_sources,
            "metric_scaling": self.metric_scaling,
            "region_of_interest": self.region_of_interest,
            "metadata": self.metadata,
        }


@dataclass
class CIKOfficialEvaluator:
    """Benchmark-faithful evaluator backed by official CiK metric code."""

    task_name: str
    future_frame: pd.DataFrame
    metric_scaling: float
    region_of_interest: Tuple[int, ...]
    metric_constraint: Any
    roi_weight: float = DEFAULT_CIK_ROI_WEIGHT
    metric_name: str = "threshold_weighted_crps.metric"

    def evaluate(self, samples: Any) -> Dict[str, float]:
        """Evaluate forecast samples with the official CiK metric implementation."""

        threshold_weighted_crps = _load_threshold_weighted_crps()
        sample_array = _normalize_forecast_samples(samples)
        target = self.future_frame.iloc[:, -1]
        region_of_interest = list(self.region_of_interest)
        return threshold_weighted_crps(
            target=target,
            forecast=sample_array,
            scaling=self.metric_scaling,
            region_of_interest=region_of_interest,
            roi_weight=self.roi_weight,
            constraint=self.metric_constraint,
            compute_variance=False,
        )


def build_cik_sample(
    row: Mapping[str, Any],
    row_index: int,
    context_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
    target_column: Optional[str | int] = None,
    include_field_names: bool = True,
) -> CIKSample:
    history_frame = parse_cik_timeframe(row["past_time"])
    future_frame = parse_cik_timeframe(row["future_time"])
    history_tensor, resolved_target_column = frame_to_target_tensor(
        history_frame, target_column=target_column
    )
    future_tensor, _ = frame_to_target_tensor(future_frame, target_column=resolved_target_column)
    context_text, context_map = merge_cik_context(
        row=row,
        context_fields=context_fields,
        include_field_names=include_field_names,
    )

    metadata = {
        "seed": row.get("seed"),
        "weight": row.get("weight"),
        "skills": tuple(row.get("skills", [])),
        "seasonal_period": row.get("seasonal_period"),
        "constraint_min": row.get("constraint_min"),
        "constraint_max": row.get("constraint_max"),
        "constraint_variable_max_index": tuple(row.get("constraint_variable_max_index", [])),
        "constraint_variable_max_values": tuple(row.get("constraint_variable_max_values", [])),
    }

    return CIKSample(
        row_index=row_index,
        task_name=str(row["name"]),
        target_column=resolved_target_column,
        history_frame=history_frame,
        future_frame=future_frame,
        history=history_tensor,
        future=future_tensor,
        context_text=context_text,
        context_fields=context_map,
        context_sources=tuple(row.get("context_sources", [])),
        metric_scaling=float(row.get("metric_scaling", 1.0)),
        region_of_interest=tuple(int(index) for index in row.get("region_of_interest", [])),
        metadata=metadata,
    )


def build_cik_official_evaluator(
    sample: CIKSample | Mapping[str, Any],
    roi_weight: float = DEFAULT_CIK_ROI_WEIGHT,
) -> CIKOfficialEvaluator:
    """Build a row-level official evaluator from a CiK sample or batch item."""

    metadata = sample.metadata if isinstance(sample, CIKSample) else sample["metadata"]
    future_frame = sample.future_frame if isinstance(sample, CIKSample) else sample["future_frame"]
    region_of_interest = (
        sample.region_of_interest if isinstance(sample, CIKSample) else sample["region_of_interest"]
    )
    metric_scaling = (
        float(sample.metric_scaling)
        if isinstance(sample, CIKSample)
        else float(sample.get("metric_scaling", metadata.get("metric_scaling", 1.0)))
    )
    task_name = sample.task_name if isinstance(sample, CIKSample) else str(sample["task_name"])
    metric_constraint = build_cik_metric_constraint(metadata)
    return CIKOfficialEvaluator(
        task_name=task_name,
        future_frame=future_frame,
        metric_scaling=metric_scaling,
        region_of_interest=tuple(int(index) for index in region_of_interest),
        metric_constraint=metric_constraint,
        roi_weight=float(roi_weight),
    )


def build_cik_metric_constraint(metadata: Mapping[str, Any]) -> Any:
    """Reconstruct official CiK constraint objects from HF row metadata."""

    ListConstraint, MaxConstraint, MinConstraint, VariableMaxConstraint = (
        _load_constraint_classes()
    )

    constraints = []
    constraint_min = float(metadata.get("constraint_min", float("-inf")))
    constraint_max = float(metadata.get("constraint_max", float("inf")))
    variable_indices = tuple(int(value) for value in metadata.get("constraint_variable_max_index", ()))
    variable_values = tuple(
        float(value) for value in metadata.get("constraint_variable_max_values", ())
    )

    if constraint_min != float("-inf") and not pd.isna(constraint_min):
        constraints.append(MinConstraint(constraint_min))
    if constraint_max != float("inf") and not pd.isna(constraint_max):
        constraints.append(MaxConstraint(constraint_max))
    if variable_indices and variable_values:
        constraints.append(
            VariableMaxConstraint(
                indices=pd.Index(variable_indices).to_numpy(dtype=int),
                thresholds=pd.Index(variable_values).to_numpy(dtype=float),
            )
        )

    if not constraints:
        return None
    if len(constraints) == 1:
        return constraints[0]
    return ListConstraint(constraints)


class CIKHFDataset(Dataset):
    def __init__(
        self,
        split: str = DEFAULT_CIK_SPLIT,
        task_name: Optional[str] = None,
        task_names: Optional[Sequence[str]] = None,
        preset_name: Optional[str] = None,
        limit: Optional[int] = None,
        limit_per_task: bool = False,
        cache_dir: Optional[str] = None,
        context_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
        target_column: Optional[str | int] = None,
        include_field_names: bool = True,
    ) -> None:
        self.rows = load_cik_rows(
            split=split,
            task_name=task_name,
            task_names=task_names,
            preset_name=preset_name,
            limit=limit,
            limit_per_task=limit_per_task,
            cache_dir=cache_dir,
        )
        self.context_fields = tuple(context_fields)
        self.target_column = target_column
        self.include_field_names = include_field_names
        self._cache: Dict[int, CIKSample] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_sample(index).to_batch_item()

    def get_sample(self, index: int) -> CIKSample:
        if index not in self._cache:
            row = self.rows[index]
            self._cache[index] = build_cik_sample(
                row=row,
                row_index=index,
                context_fields=self.context_fields,
                target_column=self.target_column,
                include_field_names=self.include_field_names,
            )
        return self._cache[index]

    def task_counts(self) -> Counter:
        return Counter(self.rows["name"])

    def summary(self) -> Dict[str, Any]:
        history_lengths: List[int] = []
        future_lengths: List[int] = []
        for sample_index in range(len(self)):
            sample = self.get_sample(sample_index)
            history_lengths.append(int(sample.history.size(0)))
            future_lengths.append(int(sample.future.size(0)))

        return {
            "rows": len(self),
            "unique_tasks": len(self.task_counts()),
            "task_counts": self.task_counts(),
            "history_min": min(history_lengths) if history_lengths else 0,
            "history_max": max(history_lengths) if history_lengths else 0,
            "future_min": min(future_lengths) if future_lengths else 0,
            "future_max": max(future_lengths) if future_lengths else 0,
        }


def cik_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty CiK batch.")

    batch_size = len(batch)
    history_channels = int(batch[0]["history"].size(-1))
    future_channels = int(batch[0]["future"].size(-1))
    max_history = max(int(item["history"].size(0)) for item in batch)
    max_future = max(int(item["future"].size(0)) for item in batch)

    histories = batch[0]["history"].new_zeros((batch_size, max_history, history_channels))
    futures = batch[0]["future"].new_zeros((batch_size, max_future, future_channels))
    history_mask = torch.zeros(batch_size, max_history, dtype=torch.bool)
    future_mask = torch.zeros(batch_size, max_future, dtype=torch.bool)

    history_lengths: List[int] = []
    future_lengths: List[int] = []

    for batch_index, item in enumerate(batch):
        history = item["history"]
        future = item["future"]
        history_length = int(history.size(0))
        future_length = int(future.size(0))

        histories[batch_index, -history_length:, :] = history
        futures[batch_index, :future_length, :] = future
        history_mask[batch_index, -history_length:] = True
        future_mask[batch_index, :future_length] = True

        history_lengths.append(history_length)
        future_lengths.append(future_length)

    collated_batch = {
        "history": histories,
        "future": futures,
        "history_mask": history_mask,
        "future_mask": future_mask,
        "history_lengths": torch.tensor(history_lengths, dtype=torch.long),
        "future_lengths": torch.tensor(future_lengths, dtype=torch.long),
        "metric_scaling": torch.tensor(
            [float(item["metric_scaling"]) for item in batch],
            dtype=torch.float32,
        ),
    }

    passthrough_keys = (
        "row_index",
        "task_name",
        "target_column",
        "history_frame",
        "future_frame",
        "history_timestamps",
        "future_timestamps",
        "context_text",
        "context_fields",
        "context_sources",
        "region_of_interest",
        "metadata",
    )
    for key in passthrough_keys:
        collated_batch[key] = [item[key] for item in batch]

    return collated_batch


def build_cik_dataloader(
    split: str = DEFAULT_CIK_SPLIT,
    task_name: Optional[str] = None,
    task_names: Optional[Sequence[str]] = None,
    preset_name: Optional[str] = None,
    limit: Optional[int] = None,
    limit_per_task: bool = False,
    cache_dir: Optional[str] = None,
    context_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
    target_column: Optional[str | int] = None,
    include_field_names: bool = True,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Tuple[CIKHFDataset, DataLoader]:
    dataset = CIKHFDataset(
        split=split,
        task_name=task_name,
        task_names=task_names,
        preset_name=preset_name,
        limit=limit,
        limit_per_task=limit_per_task,
        cache_dir=cache_dir,
        context_fields=context_fields,
        target_column=target_column,
        include_field_names=include_field_names,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=cik_collate_fn,
    )
    return dataset, dataloader


def _ensure_cik_benchmark_on_path() -> None:
    benchmark_root = Path(__file__).resolve().parents[2] / "context-is-key-forecasting"
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))


def _load_threshold_weighted_crps():
    global _OFFICIAL_THRESHOLD_WEIGHTED_CRPS
    if _OFFICIAL_THRESHOLD_WEIGHTED_CRPS is not None:
        return _OFFICIAL_THRESHOLD_WEIGHTED_CRPS

    roi_metric_module = _load_official_metrics_module("roi_metric")
    _OFFICIAL_THRESHOLD_WEIGHTED_CRPS = roi_metric_module.threshold_weighted_crps
    return _OFFICIAL_THRESHOLD_WEIGHTED_CRPS


def _load_constraint_classes():
    global _OFFICIAL_CONSTRAINT_CLASSES
    if _OFFICIAL_CONSTRAINT_CLASSES is not None:
        return _OFFICIAL_CONSTRAINT_CLASSES

    constraint_module = _load_official_metrics_module("constraints")
    _OFFICIAL_CONSTRAINT_CLASSES = (
        constraint_module.ListConstraint,
        constraint_module.MaxConstraint,
        constraint_module.MinConstraint,
        constraint_module.VariableMaxConstraint,
    )
    return _OFFICIAL_CONSTRAINT_CLASSES


def _normalize_forecast_samples(samples: Any) -> Any:
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    if hasattr(samples, "numpy") and not isinstance(samples, pd.DataFrame):
        samples = samples.numpy()
    if len(samples.shape) == 3:
        samples = samples[:, :, 0]
    return samples


def _load_official_metrics_module(module_basename: str):
    metrics_root = _get_cik_metrics_root()
    _ensure_fake_metric_package(metrics_root)

    if module_basename in ("constraints", "crps"):
        module_name = f"{_OFFICIAL_METRIC_PACKAGE}.{module_basename}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        module_path = metrics_root / f"{module_basename}.py"
        return _load_module_from_file(module_name=module_name, file_path=module_path)

    if module_basename == "roi_metric":
        _load_official_metrics_module("constraints")
        _load_official_metrics_module("crps")
        module_name = f"{_OFFICIAL_METRIC_PACKAGE}.roi_metric"
        if module_name in sys.modules:
            return sys.modules[module_name]
        module_path = metrics_root / "roi_metric.py"
        source = module_path.read_text(encoding="utf-8")
        source = "from __future__ import annotations\n" + source
        module = types.ModuleType(module_name)
        module.__file__ = str(module_path)
        module.__package__ = _OFFICIAL_METRIC_PACKAGE
        sys.modules[module_name] = module
        exec(compile(source, str(module_path), "exec"), module.__dict__)
        return module

    raise KeyError(f"Unknown official metrics module '{module_basename}'.")


def _get_cik_metrics_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "context-is-key-forecasting"
        / "cik_benchmark"
        / "metrics"
    )


def _ensure_fake_metric_package(metrics_root: Path) -> None:
    if _OFFICIAL_METRIC_PACKAGE in sys.modules:
        return
    package_module = types.ModuleType(_OFFICIAL_METRIC_PACKAGE)
    package_module.__path__ = [str(metrics_root)]
    sys.modules[_OFFICIAL_METRIC_PACKAGE] = package_module


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{module_name}' from '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_task_names(
    task_name: Optional[str] = None,
    task_names: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    normalized: List[str] = []
    if task_name:
        normalized.append(task_name)
    if task_names:
        normalized.extend(name for name in task_names if name)
    return tuple(dict.fromkeys(normalized))


__all__ = [
    "CIK_DATASET_ID",
    "DEFAULT_CIK_SPLIT",
    "DEFAULT_CONTEXT_FIELDS",
    "STIC_CIK_TASK_PRESETS",
    "CIKSample",
    "CIKHFDataset",
    "CIKOfficialEvaluator",
    "DEFAULT_CIK_ROI_WEIGHT",
    "build_cik_dataloader",
    "build_cik_metric_constraint",
    "build_cik_official_evaluator",
    "build_cik_sample",
    "cik_collate_fn",
    "frame_to_target_tensor",
    "get_cik_task_preset",
    "list_cik_task_names",
    "list_cik_task_presets",
    "load_cik_rows",
    "merge_cik_context",
    "parse_cik_timeframe",
    "resolve_cik_cache_dir",
    "resolve_cik_task_names",
    "select_cik_target_column",
]
