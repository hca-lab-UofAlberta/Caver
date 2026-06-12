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
from tiny_mlp_artifact import serialize_torch_linear_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the proposal-side Stage-E DR calibrator: width-256 GELU trunk with mean and scale heads."
    )
    parser.add_argument("--dataset-path", required=True, help="Candidate-level DR dataset JSONL.")
    parser.add_argument("--output-path", required=True, help="Output JSON path for the fitted calibrator.")
    parser.add_argument("--summary-path", required=True, help="Output JSON path for the training summary.")
    parser.add_argument(
        "--target-key",
        default="dr_pseudo_outcome",
        choices=("dr_pseudo_outcome", "dr_pseudo_outcome_clipped"),
        help="Dataset target used for calibrator fitting (default: raw dr_pseudo_outcome).",
    )
    parser.add_argument("--model-id", default=None, help="Optional explicit model id written into the artifact.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs (default: 50).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4).")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size (default: 256).")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Shared trunk width (default: 256).")
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
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience in epochs (default: 8).")
    parser.add_argument(
        "--scale-floor",
        type=float,
        default=0.02,
        help="Lower bound for the heteroscedastic scale head (default: 0.02).",
    )
    parser.add_argument(
        "--scale-ceiling",
        type=float,
        default=0.5,
        help="Upper bound for the heteroscedastic scale head (default: 0.5).",
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


def evaluate_model(*, model: Any, features: Any, targets: Any, sample_weights: Any, torch: Any) -> dict[str, float]:
    with torch.no_grad():
        raw_mean, scale, _ = model(features)
        weights = sample_weights.view(-1)
        total_weight = float(weights.sum().item())
        residual = raw_mean.view(-1) - targets.view(-1)
        mse = float(((residual ** 2) * weights).sum().item() / max(total_weight, 1e-9))
        mae = float((residual.abs() * weights).sum().item() / max(total_weight, 1e-9))
        clipped_mean = raw_mean.view(-1).clamp(0.0, 1.0)
        clipped_residual = clipped_mean - targets.view(-1)
        clipped_mse = float(((clipped_residual ** 2) * weights).sum().item() / max(total_weight, 1e-9))
        mean_scale = float((scale.view(-1) * weights).sum().item() / max(total_weight, 1e-9))
        target_mean = float((targets.view(-1) * weights).sum().item() / max(total_weight, 1e-9))
    return {
        "count": float(features.shape[0]),
        "target_mean": target_mean,
        "weighted_mse": mse,
        "weighted_rmse": math.sqrt(max(0.0, mse)),
        "weighted_mae": mae,
        "weighted_clipped_mse": clipped_mse,
        "weighted_clipped_rmse": math.sqrt(max(0.0, clipped_mse)),
        "mean_scale": mean_scale,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "error: torch is required for proposal-side DR calibrator fitting; run this via "
            "scripts/env/with_openpi_pistepnft_libero_train.sh"
        ) from exc

    if float(args.scale_ceiling) <= float(args.scale_floor):
        raise ValueError("--scale-ceiling must be greater than --scale-floor")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

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
                "sample_weight": float(
                    record.get("sample_weight", 1.0 / max(1.0, float(record["safe_candidate_count"])))
                ),
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

    def build_tensor_batch(examples_subset: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
        x_tensor = torch.tensor([example["x"] for example in examples_subset], dtype=torch.float32)
        y_tensor = torch.tensor([example["y"] for example in examples_subset], dtype=torch.float32).view(-1, 1)
        weights = torch.tensor([example["sample_weight"] for example in examples_subset], dtype=torch.float32).view(-1, 1)
        return x_tensor, y_tensor, weights

    train_x, train_y, train_w = build_tensor_batch(train_examples)
    val_x, val_y, val_w = build_tensor_batch(val_examples)
    input_dim = int(train_x.shape[1])
    scale_floor = float(args.scale_floor)
    scale_ceiling = float(args.scale_ceiling)

    class CalibratorNet(torch.nn.Module):
        def __init__(self, feature_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.trunk_linear1 = torch.nn.Linear(feature_dim, hidden_dim)
            self.trunk_linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.mean_head = torch.nn.Linear(hidden_dim, 1)
            self.scale_head = torch.nn.Linear(hidden_dim, 1)

        def forward(self, features: Any) -> tuple[Any, Any, Any]:
            hidden = torch.nn.functional.gelu(self.trunk_linear1(features))
            hidden = torch.nn.functional.gelu(self.trunk_linear2(hidden))
            raw_mean = self.mean_head(hidden)
            raw_scale = self.scale_head(hidden)
            scale = scale_floor + ((scale_ceiling - scale_floor) * torch.sigmoid(raw_scale))
            return raw_mean, scale, raw_scale

    model = CalibratorNet(input_dim, int(args.hidden_dim))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    best_epoch = 0
    best_val_loss = float("inf")
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
            raw_mean, scale, _ = model(batch_x)
            variance = scale ** 2
            per_example_loss = (((batch_y - raw_mean) ** 2) / (2.0 * variance)) + (0.5 * torch.log(variance))
            loss = (per_example_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_raw_mean, train_scale, _ = model(train_x)
            train_variance = train_scale ** 2
            train_loss = (
                (
                    ((((train_y - train_raw_mean) ** 2) / (2.0 * train_variance)) + (0.5 * torch.log(train_variance)))
                    * train_w
                ).sum().item()
                / max(float(train_w.sum().item()), 1e-9)
            )
            val_raw_mean, val_scale, _ = model(val_x)
            val_variance = val_scale ** 2
            val_loss = (
                (
                    ((((val_y - val_raw_mean) ** 2) / (2.0 * val_variance)) + (0.5 * torch.log(val_variance)))
                    * val_w
                ).sum().item()
                / max(float(val_w.sum().item()), 1e-9)
            )
        train_metrics = evaluate_model(model=model, features=train_x, targets=train_y, sample_weights=train_w, torch=torch)
        val_metrics = evaluate_model(model=model, features=val_x, targets=val_y, sample_weights=val_w, torch=torch)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse": train_metrics["weighted_rmse"],
                "val_rmse": val_metrics["weighted_rmse"],
                "train_scale": train_metrics["mean_scale"],
                "val_scale": val_metrics["mean_scale"],
            }
        )
        if val_loss < (best_val_loss - 1e-6):
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= int(args.patience):
            break

    if best_state is None:
        raise RuntimeError("DR calibrator training did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_train_metrics = evaluate_model(model=model, features=train_x, targets=train_y, sample_weights=train_w, torch=torch)
    best_val_metrics = evaluate_model(model=model, features=val_x, targets=val_y, sample_weights=val_w, torch=torch)

    model_payload = {
        "model_format": STAGEE_DR_CALIBRATOR_MODEL_FORMAT,
        "model_id": str(args.model_id or STAGEE_DR_CALIBRATOR_MODEL_ID),
        "dataset_path": str(dataset_path),
        "target_key": args.target_key,
        "mean_output": "raw_unclipped",
        "feature_spec": {
            "family_ids": family_ids,
            "policy_query_index_scale": policy_query_index_scale,
            "include_base_progress_interaction": bool(args.include_base_progress_interaction),
            "include_base_progress_sq_interaction": bool(args.include_base_progress_sq_interaction),
            "include_policy_query_index": True,
            "include_policy_query_index_sq": bool(args.include_policy_query_index_sq),
            "include_family_one_hot": True,
            "include_family_progress_interaction": bool(args.include_family_progress_interaction),
            "include_family_progress_sq_interaction": bool(args.include_family_progress_sq_interaction),
        },
        "standardization": {"mean": train_mean, "std": train_std},
        "network": {
            "trunk_layers": [
                serialize_torch_linear_layer(model.trunk_linear1, activation="gelu"),
                serialize_torch_linear_layer(model.trunk_linear2, activation="gelu"),
            ],
            "mean_head_layers": [serialize_torch_linear_layer(model.mean_head, activation="identity")],
            "scale_head_layers": [serialize_torch_linear_layer(model.scale_head, activation="identity")],
            "hidden_dim": int(args.hidden_dim),
        },
        "scale_bounds": {
            "floor": scale_floor,
            "ceiling": scale_ceiling,
        },
    }
    summary_payload = {
        "workflow": "fit_stagee_dr_calibrator_mlp_v2",
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
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "scale_floor": scale_floor,
        "scale_ceiling": scale_ceiling,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "training_history": history,
    }

    write_json(output_path, model_payload)
    write_json(summary_path, summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
