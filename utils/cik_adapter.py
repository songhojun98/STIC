from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

CIK_DATASET_ID = "ServiceNow/context-is-key"
DEFAULT_CIK_SPLIT = "test"
DEFAULT_CONTEXT_FIELDS = ("scenario", "background", "constraints")
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
        dataset = dataset.select(range(min(int(limit), len(dataset))))

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


class CIKHFDataset(Dataset):
    def __init__(
        self,
        split: str = DEFAULT_CIK_SPLIT,
        task_name: Optional[str] = None,
        task_names: Optional[Sequence[str]] = None,
        preset_name: Optional[str] = None,
        limit: Optional[int] = None,
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
    "build_cik_dataloader",
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
