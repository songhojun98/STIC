from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cik_adapter import (  # noqa: E402
    DEFAULT_CONTEXT_FIELDS,
    build_cik_dataloader,
    list_cik_task_presets,
    list_cik_task_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the official CiK Hugging Face dataset from this TSLib workspace."
    )
    parser.add_argument("--split", type=str, default="test", help="HF split name")
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Optional task name filter, e.g. ElectricityIncreaseInPredictionTask",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Optional multi-task filter",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Optional named task preset, e.g. stic_minimal or stic_all_recommended",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of rows to load after filtering",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used for the padded collate smoke test",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--context-fields",
        nargs="+",
        default=list(DEFAULT_CONTEXT_FIELDS),
        help="Context fields to merge into one text string",
    )
    parser.add_argument(
        "--show-text-chars",
        type=int,
        default=240,
        help="Maximum number of text characters to print for the first sample",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available task names for the split and exit",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List the built-in STIC-oriented CiK task presets and exit",
    )
    return parser.parse_args()


def format_counter(counter_obj, max_items: int = 10) -> str:
    items = list(counter_obj.items())[:max_items]
    return ", ".join(f"{name}:{count}" for name, count in items)


def main() -> None:
    args = parse_args()

    if args.list_tasks:
        for task_name in list_cik_task_names(split=args.split, cache_dir=args.cache_dir):
            print(task_name)
        return

    if args.list_presets:
        for preset_name, task_names in list_cik_task_presets().items():
            print(f"{preset_name}: {', '.join(task_names)}")
        return

    dataset, dataloader = build_cik_dataloader(
        split=args.split,
        task_name=args.task_name,
        task_names=args.task_names,
        preset_name=args.preset,
        limit=args.limit,
        cache_dir=args.cache_dir,
        context_fields=args.context_fields,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    if len(dataset) == 0:
        raise SystemExit("No CiK rows matched the requested filter.")

    summary = dataset.summary()
    first_sample = dataset.get_sample(0)
    batch = next(iter(dataloader))
    text_preview = first_sample.context_text[: args.show_text_chars]

    print(f"dataset_id={dataset.rows.info.dataset_name or 'ServiceNow/context-is-key'}")
    print(f"split={args.split}")
    print(f"preset={args.preset}")
    print(f"rows={summary['rows']}")
    print(f"unique_tasks={summary['unique_tasks']}")
    print(f"task_counts={format_counter(summary['task_counts'])}")
    print(f"history_len_range=({summary['history_min']}, {summary['history_max']})")
    print(f"future_len_range=({summary['future_min']}, {summary['future_max']})")
    print()
    print(f"first_task={first_sample.task_name}")
    print(f"target_column={first_sample.target_column}")
    print(f"history_shape={tuple(first_sample.history.shape)}")
    print(f"future_shape={tuple(first_sample.future.shape)}")
    print(f"metric_scaling={first_sample.metric_scaling}")
    print(f"region_of_interest={list(first_sample.region_of_interest)}")
    print(f"context_sources={list(first_sample.context_sources)}")
    print(f"context_fields={sorted(first_sample.context_fields.keys())}")
    print(f"context_preview={text_preview!r}")
    print(f"history_tail={first_sample.history[-3:, 0].tolist()}")
    print(f"future_head={first_sample.future[:3, 0].tolist()}")
    print()
    print(f"batch_history_shape={tuple(batch['history'].shape)}")
    print(f"batch_future_shape={tuple(batch['future'].shape)}")
    print(f"batch_history_mask_shape={tuple(batch['history_mask'].shape)}")
    print(f"batch_future_mask_shape={tuple(batch['future_mask'].shape)}")
    print(f"batch_history_lengths={batch['history_lengths'].tolist()}")
    print(f"batch_future_lengths={batch['future_lengths'].tolist()}")


if __name__ == "__main__":
    main()
