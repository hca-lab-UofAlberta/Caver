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
from stage0_value_proxy import stable_log_loss
from stage0_value_proxy import stable_sigmoid
from stage0_value_proxy import standardize_feature_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight Stage-0 logistic value proxy from the compact dataset.")
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
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional explicit model id written into the fitted artifact.",
    )
    parser.add_argument("--epochs", type=int, default=250, help="Maximum training epochs (default: 250).")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Initial learning rate (default: 0.05).")
    parser.add_argument("--learning-rate-decay", type=float, default=0.99, help="Per-epoch decay (default: 0.99).")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 coefficient (default: 1e-3).")
    parser.add_argument(
        "--max-positive-class-weight",
        type=float,
        default=8.0,
        help="Upper bound for positive-class reweighting (default: 8.0).",
    )
    parser.add_argument(
        "--val-modulus",
        type=int,
        default=5,
        help="Deterministic context-id hash modulus for train/val splitting (default: 5).",
    )
    parser.add_argument("--val-fold", type=int, default=0, help="Held-out fold index within --val-modulus (default: 0).")
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
    parser.add_argument("--seed", type=int, default=7, help="Training RNG seed for reproducibility (default: 7).")
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early-stopping patience measured in epochs without val-loss improvement (default: 40).",
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


def dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, features))


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


def evaluate_examples(
    examples: list[dict[str, Any]],
    *,
    weights: list[float],
    bias: float,
) -> dict[str, float]:
    if not examples:
        return {
            "count": 0.0,
            "positive_rate": 0.0,
            "weighted_log_loss": 0.0,
            "brier": 0.0,
            "accuracy": 0.0,
        }

    total_weight = 0.0
    weighted_loss = 0.0
    weighted_brier = 0.0
    weighted_correct = 0.0
    weighted_positive = 0.0

    for example in examples:
        probability = stable_sigmoid(bias + dot(weights, example["x"]))
        target = float(example["y"])
        sample_weight = float(example["sample_weight"])
        total_weight += sample_weight
        weighted_positive += sample_weight * target
        weighted_loss += sample_weight * stable_log_loss(target, probability)
        weighted_brier += sample_weight * ((probability - target) ** 2)
        weighted_correct += sample_weight * int((probability >= 0.5) == (target >= 0.5))

    return {
        "count": float(len(examples)),
        "positive_rate": weighted_positive / total_weight,
        "weighted_log_loss": weighted_loss / total_weight,
        "brier": weighted_brier / total_weight,
        "accuracy": weighted_correct / total_weight,
    }


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
            return "stage0_chunk_success_logreg_v1"
    if target_mode == "context_success_progress_v1":
        return "stage0_context_success_progress_logreg_v1"
    if target_mode == "context_success_progress_sq_v1":
        return "stage0_context_success_progress_sq_logreg_v1"
    raise ValueError(f"unsupported target mode: {target_mode}")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

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
                "sample_weight": 1.0 / float(record["context_trace_count"]),
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

    dimension = len(train_examples[0]["x"])
    weights = [0.0] * dimension
    bias = 0.0
    learning_rate = float(args.learning_rate)
    best_epoch = 0
    best_val_loss = float("inf")
    best_weights = list(weights)
    best_bias = bias
    training_history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_examples)
        grad_weights = [0.0] * dimension
        grad_bias = 0.0
        total_weight = 0.0

        for example in train_examples:
            target = float(example["y"])
            sample_weight = float(example["sample_weight"])
            sample_weight *= 1.0 + ((class_weight_positive - 1.0) * target)
            probability = stable_sigmoid(bias + dot(weights, example["x"]))
            delta = sample_weight * (probability - target)
            grad_bias += delta
            for index, feature_value in enumerate(example["x"]):
                grad_weights[index] += delta * feature_value
            total_weight += sample_weight

        scale = 1.0 / total_weight
        grad_bias *= scale
        for index in range(dimension):
            grad_weights[index] = (grad_weights[index] * scale) + (args.l2 * weights[index])
            weights[index] -= learning_rate * grad_weights[index]
        bias -= learning_rate * grad_bias

        train_metrics = evaluate_examples(train_examples, weights=weights, bias=bias)
        val_metrics = evaluate_examples(val_examples, weights=weights, bias=bias)
        training_history.append(
            {
                "epoch": float(epoch),
                "learning_rate": learning_rate,
                "train_log_loss": train_metrics["weighted_log_loss"],
                "val_log_loss": val_metrics["weighted_log_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["weighted_log_loss"] < (best_val_loss - 1e-6):
            best_val_loss = val_metrics["weighted_log_loss"]
            best_epoch = epoch
            best_weights = list(weights)
            best_bias = bias
        elif epoch - best_epoch >= args.patience:
            break

        learning_rate *= float(args.learning_rate_decay)

    final_train_metrics = evaluate_examples(train_examples, weights=best_weights, bias=best_bias)
    final_val_metrics = evaluate_examples(val_examples, weights=best_weights, bias=best_bias)

    model_payload = {
        "model_format": STAGE0_VALUE_PROXY_MODEL_FORMAT,
        "model_id": (args.model_id or derive_model_id(label_key=args.label_key, target_mode=args.target_mode)),
        "label_key": args.label_key,
        "target_mode": args.target_mode,
        "dataset_path": dataset_path,
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
            "feature_dim": dimension,
        },
        "standardization": {
            "mean": train_mean,
            "std": train_std,
        },
        "weights": best_weights,
        "bias": best_bias,
        "train_metrics": final_train_metrics,
        "val_metrics": final_val_metrics,
        "class_weight_positive": class_weight_positive,
        "raw_positive_class_weight": raw_positive_class_weight,
        "split": {
            "val_modulus": args.val_modulus,
            "val_fold": args.val_fold,
        },
    }
    write_json(output_path, model_payload)

    summary_payload = {
        "output_path": output_path,
        "summary_path": summary_path,
        "dataset_path": dataset_path,
        "label_key": args.label_key,
        "target_mode": args.target_mode,
        "epochs_requested": args.epochs,
        "epochs_completed": len(training_history),
        "best_epoch": best_epoch,
        "learning_rate_initial": args.learning_rate,
        "learning_rate_decay": args.learning_rate_decay,
        "l2": args.l2,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_metrics": final_train_metrics,
        "val_metrics": final_val_metrics,
        "feature_spec": model_payload["feature_spec"],
        "class_weight_positive": class_weight_positive,
        "raw_positive_class_weight": raw_positive_class_weight,
        "history_tail": training_history[-10:],
    }
    write_json(summary_path, summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
