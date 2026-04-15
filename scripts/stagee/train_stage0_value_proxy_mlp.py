#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from stage0_value_proxy import STAGE0_VALUE_PROXY_MODEL_FORMAT
from stage0_value_proxy import STAGE0_VALUE_PROXY_MODEL_ID
from stage0_value_proxy import build_value_proxy_feature_vector
from stage0_value_proxy import context_bucket
from stage0_value_proxy import standardize_feature_vector
from tiny_mlp_artifact import serialize_torch_linear_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the proposal-side Stage-0 value proxy: a width-256 GELU trunk with 3 bootstrap heads."
    )
    parser.add_argument("--dataset-path", required=True, help="Dataset JSONL produced by build_stage0_value_proxy_dataset.py.")
    parser.add_argument("--output-path", required=True, help="Output JSON path for the fitted model.")
    parser.add_argument("--summary-path", required=True, help="Output JSON path for the training summary.")
    parser.add_argument(
        "--label-key",
        default="context_success_label",
        choices=("context_success_label", "chunk_success_label"),
        help="Supervision target (default: context_success_label).",
    )
    parser.add_argument(
        "--target-mode",
        default="label_key",
        choices=("label_key", "context_success_progress_v1", "context_success_progress_sq_v1"),
        help="Target construction rule (default: label_key).",
    )
    parser.add_argument("--model-id", default=None, help="Optional explicit model id written into the artifact.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs (default: 30).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4).")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size (default: 256).")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Shared trunk width (default: 256).")
    parser.add_argument(
        "--bootstrap-keep-probability",
        type=float,
        default=0.8,
        help="Bernoulli keep probability for each bootstrap head and example (default: 0.8).",
    )
    parser.add_argument(
        "--max-positive-class-weight",
        type=float,
        default=8.0,
        help="Upper bound for positive-target reweighting (default: 8.0).",
    )
    parser.add_argument("--val-modulus", type=int, default=5, help="Deterministic context-id hash modulus.")
    parser.add_argument("--val-fold", type=int, default=0, help="Held-out fold index within --val-modulus.")
    parser.add_argument(
        "--policy-query-index-scale",
        type=float,
        default=None,
        help="Optional explicit scale for the policy-query feature; defaults to dataset max policy_query_index or 1.",
    )
    parser.add_argument(
        "--include-base-progress-interaction",
        action="store_true",
        help="Append base-feature times normalized progress interaction features.",
    )
    parser.add_argument(
        "--include-base-progress-sq-interaction",
        action="store_true",
        help="Append base-feature times squared normalized progress interaction features.",
    )
    parser.add_argument(
        "--include-policy-query-index-sq",
        action="store_true",
        help="Append squared normalized policy-query progress to the feature vector.",
    )
    parser.add_argument(
        "--include-family-progress-interaction",
        action="store_true",
        help="Append family-specific normalized progress interaction features.",
    )
    parser.add_argument(
        "--include-family-progress-sq-interaction",
        action="store_true",
        help="Append family-specific squared-progress interaction features.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Training RNG seed (default: 7).")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs (default: 5).")
    parser.add_argument(
        "--pre-scale-floor",
        type=float,
        default=1e-6,
        help="Minimum pre-calibration ensemble scale written into the artifact (default: 1e-6).",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to decode {path} line {line_number}") from exc
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def compute_mean_std(feature_rows: list[list[float]]) -> tuple[list[float], list[float]]:
    dimension = len(feature_rows[0])
    means = [0.0] * dimension
    for row in feature_rows:
        for index, value in enumerate(row):
            means[index] += value
    count = float(len(feature_rows))
    means = [value / count for value in means]

    stds = [0.0] * dimension
    for row in feature_rows:
        for index, value in enumerate(row):
            stds[index] += (value - means[index]) ** 2
    stds = [math.sqrt(value / count) if value > 1e-12 else 1.0 for value in stds]
    return means, stds


def resolve_target(record: dict[str, Any], *, label_key: str, target_mode: str) -> float:
    if target_mode == "label_key":
        return float(record[label_key])
    progress = (float(record["policy_query_index"]) + 1.0) / max(1.0, float(record["context_trace_count"]))
    context_success = float(record["context_success_label"])
    if target_mode == "context_success_progress_v1":
        return context_success * progress
    if target_mode == "context_success_progress_sq_v1":
        return context_success * (progress ** 2)
    raise ValueError(f"unsupported target mode: {target_mode}")


def derive_model_id(*, label_key: str, target_mode: str) -> str:
    if target_mode == "label_key":
        if label_key == "context_success_label":
            return STAGE0_VALUE_PROXY_MODEL_ID
        if label_key == "chunk_success_label":
            return "stage0_chunk_success_mlp3head_v2"
    if target_mode == "context_success_progress_v1":
        return "stage0_context_success_progress_mlp3head_v2"
    if target_mode == "context_success_progress_sq_v1":
        return "stage0_context_success_progress_sq_mlp3head_v2"
    raise ValueError(f"unsupported target mode: {target_mode}")


def evaluate_model(*, model: Any, features: Any, targets: Any, sample_weights: Any, torch: Any) -> dict[str, float]:
    with torch.no_grad():
        predictions = model(features)
        mean_prediction = predictions.mean(dim=1)
        clipped_mean = mean_prediction.clamp(0.0, 1.0)
        pre_scale = predictions.std(dim=1, unbiased=False)
        weights = sample_weights.view(-1)
        total_weight = float(weights.sum().item())
        mse = float((((clipped_mean - targets.view(-1)) ** 2) * weights).sum().item() / max(total_weight, 1e-9))
        mae = float(((clipped_mean - targets.view(-1)).abs() * weights).sum().item() / max(total_weight, 1e-9))
        target_mean = float((targets.view(-1) * weights).sum().item() / max(total_weight, 1e-9))
        mean_pre_scale = float((pre_scale * weights).sum().item() / max(total_weight, 1e-9))
    return {
        "count": float(features.shape[0]),
        "target_mean": target_mean,
        "weighted_mse": mse,
        "weighted_rmse": math.sqrt(max(0.0, mse)),
        "weighted_mae": mae,
        "mean_pre_scale": mean_pre_scale,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "error: torch is required for proposal-side value-proxy fitting; run this via "
            "scripts/env/with_openpi_pistepnft_libero_train.sh"
        ) from exc

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    dataset_path = Path(args.dataset_path).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve()

    dataset_records = read_jsonl(dataset_path)
    if not dataset_records:
        raise ValueError(f"dataset contains zero records: {dataset_path}")

    family_ids = sorted({str(record["proxy_family_id"]) for record in dataset_records})
    policy_query_index_scale = float(
        args.policy_query_index_scale
        if args.policy_query_index_scale is not None
        else max(int(record["policy_query_index"]) for record in dataset_records) or 1
    )

    examples: list[dict[str, Any]] = []
    for record in dataset_records:
        feature_vector = build_value_proxy_feature_vector(
            record["base_feature_vector"],
            proxy_family_id=record["proxy_family_id"],
            policy_query_index=int(record["policy_query_index"]),
            family_ids=family_ids,
            policy_query_index_scale=policy_query_index_scale,
            include_base_progress_interaction=bool(args.include_base_progress_interaction),
            include_base_progress_sq_interaction=bool(args.include_base_progress_sq_interaction),
            include_policy_query_index=True,
            include_policy_query_index_sq=bool(args.include_policy_query_index_sq),
            include_family_one_hot=True,
            include_family_progress_interaction=bool(args.include_family_progress_interaction),
            include_family_progress_sq_interaction=bool(args.include_family_progress_sq_interaction),
        )
        examples.append(
            {
                "context_id": str(record["context_id"]),
                "x_raw": feature_vector,
                "y": resolve_target(record, label_key=args.label_key, target_mode=args.target_mode),
                "sample_weight": 1.0 / max(1.0, float(record["context_trace_count"])),
            }
        )

    train_examples = [
        example
        for example in examples
        if context_bucket(example["context_id"], bucket_count=args.val_modulus) != args.val_fold
    ]
    val_examples = [
        example
        for example in examples
        if context_bucket(example["context_id"], bucket_count=args.val_modulus) == args.val_fold
    ]
    if not train_examples or not val_examples:
        raise ValueError("train/val split produced an empty partition")

    train_mean, train_std = compute_mean_std([example["x_raw"] for example in train_examples])
    for example in examples:
        example["x"] = standardize_feature_vector(example["x_raw"], mean=train_mean, std=train_std)

    negative_mass = sum(example["sample_weight"] * (1.0 - example["y"]) for example in train_examples)
    positive_mass = sum(example["sample_weight"] * example["y"] for example in train_examples)
    raw_positive_class_weight = (negative_mass / positive_mass) if positive_mass > 0.0 else 1.0
    class_weight_positive = min(float(args.max_positive_class_weight), raw_positive_class_weight)

    def build_tensor_batch(examples_subset: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
        x_tensor = torch.tensor([example["x"] for example in examples_subset], dtype=torch.float32)
        y_tensor = torch.tensor([example["y"] for example in examples_subset], dtype=torch.float32).view(-1, 1)
        base_weight = torch.tensor([example["sample_weight"] for example in examples_subset], dtype=torch.float32).view(-1, 1)
        class_weight = 1.0 + ((class_weight_positive - 1.0) * y_tensor)
        return x_tensor, y_tensor, (base_weight * class_weight)

    train_x, train_y, train_w = build_tensor_batch(train_examples)
    val_x, val_y, val_w = build_tensor_batch(val_examples)
    input_dim = int(train_x.shape[1])

    class ValueProxyNet(torch.nn.Module):
        def __init__(self, feature_dim: int, hidden_dim: int, head_count: int) -> None:
            super().__init__()
            self.trunk_linear1 = torch.nn.Linear(feature_dim, hidden_dim)
            self.trunk_linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, 1) for _ in range(head_count)])

        def forward(self, features: Any) -> Any:
            hidden = torch.nn.functional.gelu(self.trunk_linear1(features))
            hidden = torch.nn.functional.gelu(self.trunk_linear2(hidden))
            return torch.cat([head(hidden) for head in self.heads], dim=1)

    model = ValueProxyNet(input_dim, int(args.hidden_dim), head_count=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    best_epoch = 0
    best_val_mse = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train_x.shape[0])
        for start in range(0, int(train_x.shape[0]), int(args.batch_size)):
            indices = permutation[start : start + int(args.batch_size)]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            batch_w = train_w[indices]
            predictions = model(batch_x)
            bootstrap_mask = torch.bernoulli(
                torch.full(
                    (batch_x.shape[0], predictions.shape[1]),
                    float(args.bootstrap_keep_probability),
                    dtype=torch.float32,
                )
            )
            weighted_mask = batch_w * bootstrap_mask
            squared_error = (predictions - batch_y) ** 2
            loss = (squared_error * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(model=model, features=train_x, targets=train_y, sample_weights=train_w, torch=torch)
        val_metrics = evaluate_model(model=model, features=val_x, targets=val_y, sample_weights=val_w, torch=torch)
        history.append(
            {
                "epoch": float(epoch),
                "train_rmse": train_metrics["weighted_rmse"],
                "train_mae": train_metrics["weighted_mae"],
                "train_pre_scale": train_metrics["mean_pre_scale"],
                "val_rmse": val_metrics["weighted_rmse"],
                "val_mae": val_metrics["weighted_mae"],
                "val_pre_scale": val_metrics["mean_pre_scale"],
                "val_mse": val_metrics["weighted_mse"],
            }
        )
        if val_metrics["weighted_mse"] < (best_val_mse - 1e-6):
            best_val_mse = val_metrics["weighted_mse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= int(args.patience):
            break

    if best_state is None:
        raise RuntimeError("value proxy training did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_train_metrics = evaluate_model(model=model, features=train_x, targets=train_y, sample_weights=train_w, torch=torch)
    final_val_metrics = evaluate_model(model=model, features=val_x, targets=val_y, sample_weights=val_w, torch=torch)

    model_payload = {
        "model_format": STAGE0_VALUE_PROXY_MODEL_FORMAT,
        "model_id": (args.model_id or derive_model_id(label_key=args.label_key, target_mode=args.target_mode)),
        "dataset_path": str(dataset_path),
        "label_key": args.label_key,
        "target_mode": args.target_mode,
        "feature_spec": {
            "family_ids": family_ids,
            "base_feature_dim": len(dataset_records[0]["base_feature_vector"]),
            "include_base_progress_interaction": bool(args.include_base_progress_interaction),
            "include_base_progress_sq_interaction": bool(args.include_base_progress_sq_interaction),
            "include_policy_query_index": True,
            "include_policy_query_index_sq": bool(args.include_policy_query_index_sq),
            "include_family_one_hot": True,
            "include_family_progress_interaction": bool(args.include_family_progress_interaction),
            "include_family_progress_sq_interaction": bool(args.include_family_progress_sq_interaction),
            "policy_query_index_scale": policy_query_index_scale,
            "feature_dim": input_dim,
        },
        "standardization": {
            "mean": train_mean,
            "std": train_std,
        },
        "network": {
            "trunk_layers": [
                serialize_torch_linear_layer(model.trunk_linear1, activation="gelu"),
                serialize_torch_linear_layer(model.trunk_linear2, activation="gelu"),
            ],
            "head_layers": [
                [serialize_torch_linear_layer(head, activation="identity")]
                for head in model.heads
            ],
            "hidden_dim": int(args.hidden_dim),
            "head_count": 3,
            "bootstrap_keep_probability": float(args.bootstrap_keep_probability),
        },
        "pre_scale_floor": float(args.pre_scale_floor),
        "train_metrics": final_train_metrics,
        "val_metrics": final_val_metrics,
        "class_weight_positive": class_weight_positive,
        "raw_positive_class_weight": raw_positive_class_weight,
        "split": {
            "val_modulus": args.val_modulus,
            "val_fold": args.val_fold,
        },
    }
    summary_payload = {
        "workflow": "train_stage0_value_proxy_mlp_v2",
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "label_key": args.label_key,
        "target_mode": args.target_mode,
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "bootstrap_keep_probability": float(args.bootstrap_keep_probability),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "feature_spec": model_payload["feature_spec"],
        "class_weight_positive": class_weight_positive,
        "raw_positive_class_weight": raw_positive_class_weight,
        "best_train_metrics": final_train_metrics,
        "best_val_metrics": final_val_metrics,
        "history_tail": history[-10:],
    }

    write_json(output_path, model_payload)
    write_json(summary_path, summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
