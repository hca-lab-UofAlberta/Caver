#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
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
from stagee_dr_calibration import STAGEE_DR_DATASET_FORMAT
from stagee_lvd_selector import STAGEE_LVD_SELECTOR_MODEL_FORMAT
from stagee_lvd_selector import STAGEE_LVD_SELECTOR_MODEL_ID
from tiny_mlp_artifact import serialize_torch_linear_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a Stage-E CAVER-LVD listwise selector from candidate-level DR datasets."
    )
    parser.add_argument("--dataset-path", action="append", required=True, help="Candidate-level DR dataset JSONL.")
    parser.add_argument("--output-path", required=True, help="Output JSON path for the LVD selector artifact.")
    parser.add_argument("--summary-path", required=True, help="Output JSON path for the training summary.")
    parser.add_argument(
        "--target-source",
        default="dr_clipped",
        choices=("dr_clipped", "dr_raw", "observed_selected_else_nuisance", "nuisance"),
        help="Pseudo-outcome source used for listwise targets.",
    )
    parser.add_argument("--model-id", default=None, help="Optional explicit model id written into the artifact.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-temperature", type=float, default=0.20)
    parser.add_argument("--train-temperature", type=float, default=0.50)
    parser.add_argument("--selector-temperature", type=float, default=0.50)
    parser.add_argument("--exploration-floor", type=float, default=0.10)
    parser.add_argument("--val-modulus", type=int, default=5)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--policy-query-index-scale", type=float, default=None)
    parser.add_argument("--include-base-progress-interaction", action="store_true")
    parser.add_argument("--include-base-progress-sq-interaction", action="store_true")
    parser.add_argument("--include-policy-query-index-sq", action="store_true")
    parser.add_argument("--include-family-progress-interaction", action="store_true")
    parser.add_argument("--include-family-progress-sq-interaction", action="store_true")
    parser.add_argument(
        "--min-candidates-per-group",
        type=int,
        default=2,
        help="Drop candidate menus smaller than this value; LVD needs at least two candidates.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to decode {path} line {line_number}") from exc
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compute_mean_std(feature_rows: list[list[float]]) -> tuple[list[float], list[float]]:
    dimension = len(feature_rows[0])
    means = [0.0] * dimension
    for row in feature_rows:
        for index, value in enumerate(row):
            means[index] += value
    means = [value / float(len(feature_rows)) for value in means]

    stds = [0.0] * dimension
    for row in feature_rows:
        for index, value in enumerate(row):
            stds[index] += (value - means[index]) ** 2
    stds = [math.sqrt(value / float(len(feature_rows))) if value > 1e-12 else 1.0 for value in stds]
    return means, stds


def resolve_target(record: dict[str, Any], *, target_source: str) -> float:
    if target_source == "dr_clipped":
        value = record["dr_pseudo_outcome_clipped"]
    elif target_source == "dr_raw":
        value = record["dr_pseudo_outcome"]
    elif target_source == "nuisance":
        value = record["lagged_nuisance_mean"]
    elif target_source == "observed_selected_else_nuisance":
        value = record["executed_outcome"] if bool(record["candidate_selected"]) else record["lagged_nuisance_mean"]
    else:
        raise ValueError(f"unsupported target source: {target_source}")
    return max(0.0, min(1.0, float(value)))


def softmax(values: list[float], *, temperature: float) -> list[float]:
    if not values:
        return []
    temp = max(float(temperature), 1e-6)
    maximum = max(values) / temp
    weights = [math.exp((value / temp) - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def evaluate_groups(*, model: Any, groups: list[dict[str, Any]], torch: Any, temperature: float) -> dict[str, float]:
    if not groups:
        return {
            "groups": 0.0,
            "listwise_ce": float("nan"),
            "top1_target_match": float("nan"),
            "expected_target": float("nan"),
        }
    total_ce = 0.0
    top_match = 0
    expected_target = 0.0
    with torch.no_grad():
        for group in groups:
            x = torch.tensor(group["x"], dtype=torch.float32)
            target = torch.tensor(group["target_distribution"], dtype=torch.float32)
            y = torch.tensor(group["target_values"], dtype=torch.float32)
            logits = model(x).view(-1) / max(float(temperature), 1e-6)
            log_probs = torch.nn.functional.log_softmax(logits, dim=0)
            probs = torch.nn.functional.softmax(logits, dim=0)
            total_ce += float((-(target * log_probs)).sum().item())
            top_match += int(torch.argmax(probs).item() == torch.argmax(target).item())
            expected_target += float((probs * y).sum().item())
    return {
        "groups": float(len(groups)),
        "listwise_ce": total_ce / float(len(groups)),
        "top1_target_match": top_match / float(len(groups)),
        "expected_target": expected_target / float(len(groups)),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    if args.target_temperature <= 0.0:
        raise ValueError("--target-temperature must be positive")
    if args.train_temperature <= 0.0 or args.selector_temperature <= 0.0:
        raise ValueError("selector/train temperatures must be positive")
    if not (0.0 <= args.exploration_floor < 1.0):
        raise ValueError("--exploration-floor must be in [0, 1)")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "error: torch is required for LVD selector fitting; run via "
            "scripts/env/with_openpi_pistepnft_libero_train.sh"
        ) from exc

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    dataset_paths = [Path(path).resolve() for path in args.dataset_path]
    dataset_records: list[dict[str, Any]] = []
    for source_index, path in enumerate(dataset_paths):
        for record in read_jsonl(path):
            annotated_record = dict(record)
            annotated_record["_lvd_source_index"] = source_index
            annotated_record["_lvd_source_path"] = str(path)
            dataset_records.append(annotated_record)
    if not dataset_records:
        raise ValueError("all input datasets are empty")
    for record in dataset_records:
        if record.get("dataset_format") != STAGEE_DR_DATASET_FORMAT:
            raise ValueError(f"unexpected dataset format {record.get('dataset_format')!r}")

    grouped_raw: dict[tuple[int, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in dataset_records:
        grouped_raw[
            (
                int(record["_lvd_source_index"]),
                str(record["context_id"]),
                int(record["policy_query_index"]),
            )
        ].append(record)
    grouped_raw = {
        key: sorted(group, key=lambda item: int(item["candidate_index"]))
        for key, group in grouped_raw.items()
        if len(group) >= int(args.min_candidates_per_group)
    }
    if not grouped_raw:
        raise ValueError("LVD training requires at least one multi-candidate group")

    family_ids = sorted({str(record["proxy_family_id"]) for group in grouped_raw.values() for record in group})
    policy_query_index_scale = float(
        args.policy_query_index_scale
        if args.policy_query_index_scale is not None
        else max(int(record["policy_query_index"]) for group in grouped_raw.values() for record in group) or 1
    )

    groups: list[dict[str, Any]] = []
    for (source_index, context_id, policy_query_index), records in sorted(grouped_raw.items()):
        examples: list[dict[str, Any]] = []
        target_values: list[float] = []
        for record in records:
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
                    "candidate_index": int(record["candidate_index"]),
                    "x_raw": feature_vector,
                    "proxy_family_id": str(record["proxy_family_id"]),
                    "candidate_selected": bool(record["candidate_selected"]),
                    "logged_propensity": float(record["logged_propensity"]),
                }
            )
            target_values.append(resolve_target(record, target_source=args.target_source))
        groups.append(
            {
                "context_id": context_id,
                "split_id": f"{source_index}:{context_id}",
                "source_index": source_index,
                "policy_query_index": policy_query_index,
                "examples": examples,
                "target_values": target_values,
                "target_distribution": softmax(target_values, temperature=float(args.target_temperature)),
            }
        )

    train_groups = [
        group
        for group in groups
        if context_bucket(group["split_id"], bucket_count=args.val_modulus) != args.val_fold
    ]
    val_groups = [
        group
        for group in groups
        if context_bucket(group["split_id"], bucket_count=args.val_modulus) == args.val_fold
    ]
    split_mode = "hashed_train_val"
    if not train_groups or not val_groups:
        split_mode = "all_data_fallback"
        train_groups = list(groups)
        val_groups = list(groups)

    train_feature_rows = [example["x_raw"] for group in train_groups for example in group["examples"]]
    train_mean, train_std = compute_mean_std(train_feature_rows)
    for group in groups:
        group["x"] = [
            standardize_feature_vector(example["x_raw"], mean=train_mean, std=train_std)
            for example in group["examples"]
        ]

    input_dim = len(train_groups[0]["x"][0])

    class LVDSelectorNet(torch.nn.Module):
        def __init__(self, feature_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.trunk_linear1 = torch.nn.Linear(feature_dim, hidden_dim)
            self.trunk_linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.score_head = torch.nn.Linear(hidden_dim, 1)

        def forward(self, features: Any) -> Any:
            hidden = torch.nn.functional.gelu(self.trunk_linear1(features))
            hidden = torch.nn.functional.gelu(self.trunk_linear2(hidden))
            return self.score_head(hidden)

    model = LVDSelectorNet(input_dim, int(args.hidden_dim))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    best_epoch = 0
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        shuffled_groups = list(train_groups)
        random.shuffle(shuffled_groups)
        train_loss_sum = 0.0
        for group in shuffled_groups:
            x = torch.tensor(group["x"], dtype=torch.float32)
            target = torch.tensor(group["target_distribution"], dtype=torch.float32)
            logits = model(x).view(-1) / max(float(args.train_temperature), 1e-6)
            log_probs = torch.nn.functional.log_softmax(logits, dim=0)
            loss = -(target * log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
        train_metrics = evaluate_groups(
            model=model,
            groups=train_groups,
            torch=torch,
            temperature=float(args.train_temperature),
        )
        val_metrics = evaluate_groups(
            model=model,
            groups=val_groups,
            torch=torch,
            temperature=float(args.train_temperature),
        )
        val_loss = val_metrics["listwise_ce"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / float(len(train_groups)),
                "train_listwise_ce": train_metrics["listwise_ce"],
                "val_listwise_ce": val_loss,
                "train_top1_target_match": train_metrics["top1_target_match"],
                "val_top1_target_match": val_metrics["top1_target_match"],
                "train_expected_target": train_metrics["expected_target"],
                "val_expected_target": val_metrics["expected_target"],
            }
        )
        if val_loss < (best_val_loss - 1e-6):
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= int(args.patience):
            break

    if best_state is None:
        raise RuntimeError("LVD selector training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics = evaluate_groups(
        model=model,
        groups=train_groups,
        torch=torch,
        temperature=float(args.train_temperature),
    )
    val_metrics = evaluate_groups(
        model=model,
        groups=val_groups,
        torch=torch,
        temperature=float(args.train_temperature),
    )

    model_payload = {
        "model_format": STAGEE_LVD_SELECTOR_MODEL_FORMAT,
        "model_id": str(args.model_id or STAGEE_LVD_SELECTOR_MODEL_ID),
        "dataset_paths": [str(path) for path in dataset_paths],
        "target_source": args.target_source,
        "target_temperature": float(args.target_temperature),
        "train_temperature": float(args.train_temperature),
        "selector_temperature": float(args.selector_temperature),
        "exploration_floor": float(args.exploration_floor),
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
            "score_head_layers": [serialize_torch_linear_layer(model.score_head, activation="identity")],
            "hidden_dim": int(args.hidden_dim),
        },
    }
    summary_payload = {
        "workflow": "fit_stagee_lvd_selector_v1",
        "dataset_paths": [str(path) for path in dataset_paths],
        "output_path": str(Path(args.output_path).resolve()),
        "summary_path": str(Path(args.summary_path).resolve()),
        "target_source": args.target_source,
        "records_total": len(dataset_records),
        "groups_total": len(groups),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "family_ids": family_ids,
        "policy_query_index_scale": policy_query_index_scale,
        "split_mode": split_mode,
        "best_epoch": best_epoch,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "target_temperature": float(args.target_temperature),
        "train_temperature": float(args.train_temperature),
        "selector_temperature": float(args.selector_temperature),
        "exploration_floor": float(args.exploration_floor),
        "best_train_metrics": train_metrics,
        "best_val_metrics": val_metrics,
        "training_history": history,
    }

    write_json(Path(args.output_path).resolve(), model_payload)
    write_json(Path(args.summary_path).resolve(), summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
