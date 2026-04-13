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

from stage0_value_proxy import build_value_proxy_feature_vector
from stage0_value_proxy import context_bucket
from stage0_value_proxy import standardize_feature_vector
from stagee_dr_calibration import STAGEE_DR_CALIBRATOR_MODEL_FORMAT
from stagee_dr_calibration import STAGEE_DR_CALIBRATOR_MODEL_ID
from stagee_dr_calibration import STAGEE_DR_DATASET_FORMAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a lagged Stage-E DR calibrator from a candidate-level DR dataset.")
    parser.add_argument("--dataset-path", required=True, help="Candidate-level DR dataset JSONL.")
    parser.add_argument("--output-path", required=True, help="Output JSON path for the fitted calibrator.")
    parser.add_argument("--summary-path", required=True, help="Output JSON path for the training summary.")
    parser.add_argument(
        "--target-key",
        default="dr_pseudo_outcome_clipped",
        choices=("dr_pseudo_outcome", "dr_pseudo_outcome_clipped"),
        help="Dataset target used for calibrator fitting (default: dr_pseudo_outcome_clipped).",
    )
    parser.add_argument("--model-id", default=None, help="Optional explicit model id written into the artifact.")
    parser.add_argument("--epochs", type=int, default=250, help="Maximum training epochs (default: 250).")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Initial learning rate (default: 0.05).")
    parser.add_argument("--learning-rate-decay", type=float, default=0.99, help="Per-epoch decay (default: 0.99).")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 coefficient (default: 1e-3).")
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
    parser.add_argument("--patience", type=int, default=40, help="Early-stopping patience in epochs.")
    parser.add_argument(
        "--residual-scale-floor",
        type=float,
        default=1e-3,
        help="Minimum residual scale stored in the calibrator artifact (default: 1e-3).",
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
            "target_mean": 0.0,
            "weighted_mse": 0.0,
            "weighted_rmse": 0.0,
            "weighted_mae": 0.0,
        }

    total_weight = 0.0
    weighted_target = 0.0
    weighted_mse = 0.0
    weighted_mae = 0.0
    for example in examples:
        prediction = bias + dot(weights, example["x"])
        target = float(example["y"])
        sample_weight = float(example["sample_weight"])
        residual = prediction - target
        total_weight += sample_weight
        weighted_target += sample_weight * target
        weighted_mse += sample_weight * (residual ** 2)
        weighted_mae += sample_weight * abs(residual)

    mse = weighted_mse / total_weight
    return {
        "count": float(len(examples)),
        "target_mean": weighted_target / total_weight,
        "weighted_mse": mse,
        "weighted_rmse": math.sqrt(max(0.0, mse)),
        "weighted_mae": weighted_mae / total_weight,
    }


def compute_weighted_residual_scales(
    examples: list[dict[str, Any]],
    *,
    weights: list[float],
    bias: float,
    residual_scale_floor: float,
) -> tuple[float, dict[str, float]]:
    totals: dict[str, float] = {}
    means: dict[str, float] = {}
    sq_totals: dict[str, float] = {}
    global_total = 0.0
    global_sq_total = 0.0

    for example in examples:
        prediction = bias + dot(weights, example["x"])
        residual = float(example["y"]) - prediction
        sample_weight = float(example["sample_weight"])
        family_id = str(example["proxy_family_id"])
        global_total += sample_weight
        global_sq_total += sample_weight * (residual ** 2)
        totals[family_id] = totals.get(family_id, 0.0) + sample_weight
        sq_totals[family_id] = sq_totals.get(family_id, 0.0) + (sample_weight * (residual ** 2))
        means[family_id] = means.get(family_id, 0.0)

    global_scale = math.sqrt(global_sq_total / global_total) if global_total > 0.0 else residual_scale_floor
    global_scale = max(residual_scale_floor, global_scale)
    family_scales: dict[str, float] = {}
    for family_id, total_weight in totals.items():
        scale = math.sqrt(sq_totals[family_id] / total_weight) if total_weight > 0.0 else global_scale
        family_scales[family_id] = max(residual_scale_floor, scale)
    return global_scale, family_scales


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    dataset_path = Path(args.dataset_path).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve()

    dataset_records = read_jsonl(dataset_path)
    if not dataset_records:
        raise ValueError(f"dataset contains zero records: {dataset_path}")
    dataset_format = dataset_records[0].get("dataset_format")
    if dataset_format != STAGEE_DR_DATASET_FORMAT:
        raise ValueError(f"unexpected dataset format {dataset_format!r} in {dataset_path}")

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
                "proxy_family_id": str(record["proxy_family_id"]),
                "x_raw": feature_vector,
                "y": float(record[args.target_key]),
                "sample_weight": 1.0 / max(1.0, float(record["safe_candidate_count"])),
            }
        )

    split_mode = "hashed_train_val"
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
        split_mode = "all_data_fallback"
        train_examples = list(examples)
        val_examples = list(examples)

    train_mean, train_std = compute_mean_std([example["x_raw"] for example in train_examples])
    for example in examples:
        example["x"] = standardize_feature_vector(example["x_raw"], mean=train_mean, std=train_std)

    dimension = len(train_examples[0]["x"])
    mean_target = sum(float(example["y"]) for example in train_examples) / float(len(train_examples))
    weights = [0.0] * dimension
    bias = mean_target
    learning_rate = float(args.learning_rate)
    best_epoch = 0
    best_val_mse = float("inf")
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
            prediction = bias + dot(weights, example["x"])
            delta = sample_weight * (prediction - target)
            grad_bias += delta
            for index, feature_value in enumerate(example["x"]):
                grad_weights[index] += delta * feature_value
            total_weight += sample_weight

        scale = 1.0 / max(total_weight, 1e-9)
        grad_bias *= scale
        for index in range(dimension):
            grad_weights[index] = (grad_weights[index] * scale) + (args.l2 * weights[index])
            weights[index] -= learning_rate * grad_weights[index]
        bias -= learning_rate * grad_bias

        train_metrics = evaluate_examples(train_examples, weights=weights, bias=bias)
        val_metrics = evaluate_examples(val_examples, weights=weights, bias=bias)
        history_row = {
            "epoch": float(epoch),
            "learning_rate": learning_rate,
            "train_rmse": train_metrics["weighted_rmse"],
            "train_mae": train_metrics["weighted_mae"],
            "val_rmse": val_metrics["weighted_rmse"],
            "val_mae": val_metrics["weighted_mae"],
            "val_mse": val_metrics["weighted_mse"],
        }
        training_history.append(history_row)

        if val_metrics["weighted_mse"] < best_val_mse:
            best_epoch = epoch
            best_val_mse = val_metrics["weighted_mse"]
            best_weights = list(weights)
            best_bias = bias

        if epoch - best_epoch >= args.patience:
            break
        learning_rate *= float(args.learning_rate_decay)

    best_train_metrics = evaluate_examples(train_examples, weights=best_weights, bias=best_bias)
    best_val_metrics = evaluate_examples(val_examples, weights=best_weights, bias=best_bias)
    residual_scale_global, residual_scale_by_family = compute_weighted_residual_scales(
        train_examples,
        weights=best_weights,
        bias=best_bias,
        residual_scale_floor=float(args.residual_scale_floor),
    )

    feature_spec = {
        "family_ids": family_ids,
        "policy_query_index_scale": policy_query_index_scale,
        "include_base_progress_interaction": bool(args.include_base_progress_interaction),
        "include_base_progress_sq_interaction": bool(args.include_base_progress_sq_interaction),
        "include_policy_query_index": True,
        "include_policy_query_index_sq": bool(args.include_policy_query_index_sq),
        "include_family_one_hot": True,
        "include_family_progress_interaction": bool(args.include_family_progress_interaction),
        "include_family_progress_sq_interaction": bool(args.include_family_progress_sq_interaction),
    }
    model_payload = {
        "model_format": STAGEE_DR_CALIBRATOR_MODEL_FORMAT,
        "model_id": str(args.model_id or STAGEE_DR_CALIBRATOR_MODEL_ID),
        "dataset_path": str(dataset_path),
        "feature_spec": feature_spec,
        "standardization": {"mean": train_mean, "std": train_std},
        "bias": float(best_bias),
        "weights": [float(weight) for weight in best_weights],
        "residual_scale_global": float(residual_scale_global),
        "residual_scale_by_family": {key: float(value) for key, value in residual_scale_by_family.items()},
        "residual_scale_floor": float(args.residual_scale_floor),
    }
    summary_payload = {
        "workflow": "fit_stagee_dr_calibrator_v1",
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "target_key": args.target_key,
        "records_total": len(dataset_records),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "family_ids": family_ids,
        "policy_query_index_scale": policy_query_index_scale,
        "split_mode": split_mode,
        "best_epoch": best_epoch,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "residual_scale_global": float(residual_scale_global),
        "residual_scale_by_family": {key: float(value) for key, value in residual_scale_by_family.items()},
        "training_history": training_history,
    }

    write_json(output_path, model_payload)
    write_json(summary_path, summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
