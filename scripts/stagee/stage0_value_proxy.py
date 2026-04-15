from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from typing import Sequence

from tiny_mlp_artifact import clamp_unit_interval
from tiny_mlp_artifact import forward_stack
from tiny_mlp_artifact import mean_and_population_std
from tiny_mlp_artifact import stable_sigmoid


STAGE0_VALUE_PROXY_MODEL_FORMAT_LEGACY = "stage0_value_proxy_logreg_v1"
STAGE0_VALUE_PROXY_MODEL_FORMAT = "stage0_value_proxy_mlp3head_v2"
STAGE0_VALUE_PROXY_SUPPORTED_MODEL_FORMATS = {
    STAGE0_VALUE_PROXY_MODEL_FORMAT_LEGACY,
    STAGE0_VALUE_PROXY_MODEL_FORMAT,
}
STAGE0_VALUE_PROXY_MODEL_ID_LEGACY = "stage0_context_success_logreg_v1"
STAGE0_VALUE_PROXY_MODEL_ID = "stage0_context_success_mlp3head_v2"


def stable_log_loss(target: float, probability: float) -> float:
    clipped = min(max(probability, 1e-8), 1.0 - 1e-8)
    target_value = min(max(float(target), 0.0), 1.0)
    return (-target_value * math.log(clipped)) - ((1.0 - target_value) * math.log(1.0 - clipped))


def context_bucket(context_id: str, *, bucket_count: int) -> int:
    digest = hashlib.sha1(context_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % bucket_count


def build_value_proxy_feature_vector(
    base_feature_vector: Sequence[float],
    *,
    proxy_family_id: str | None,
    policy_query_index: int,
    family_ids: Sequence[str],
    policy_query_index_scale: float,
    include_policy_query_index: bool = True,
    include_policy_query_index_sq: bool = False,
    include_family_one_hot: bool = True,
    include_family_progress_interaction: bool = False,
    include_family_progress_sq_interaction: bool = False,
    include_base_progress_interaction: bool = False,
    include_base_progress_sq_interaction: bool = False,
) -> list[float]:
    base_features = [float(value) for value in base_feature_vector]
    features = list(base_features)
    query_scale = policy_query_index_scale if policy_query_index_scale > 0.0 else 1.0
    capped_index = min(max(float(policy_query_index), 0.0), query_scale)
    normalized_progress = capped_index / query_scale
    normalized_progress_sq = normalized_progress * normalized_progress
    if include_base_progress_interaction:
        features.extend(normalized_progress * value for value in base_features)
    if include_base_progress_sq_interaction:
        features.extend(normalized_progress_sq * value for value in base_features)
    if include_policy_query_index:
        features.append(normalized_progress)
    if include_policy_query_index_sq:
        features.append(normalized_progress_sq)
    if include_family_one_hot:
        for family_id in family_ids:
            features.append(1.0 if proxy_family_id == family_id else 0.0)
    if include_family_progress_interaction:
        for family_id in family_ids:
            features.append(normalized_progress if proxy_family_id == family_id else 0.0)
    if include_family_progress_sq_interaction:
        for family_id in family_ids:
            features.append(normalized_progress_sq if proxy_family_id == family_id else 0.0)
    return features


def standardize_feature_vector(
    feature_vector: Sequence[float],
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> list[float]:
    return [
        (float(value) - float(feature_mean)) / float(feature_std)
        for value, feature_mean, feature_std in zip(feature_vector, mean, std)
    ]


def _legacy_predict_value_proxy(model: dict[str, Any], standardized_feature_vector: Sequence[float]) -> dict[str, Any]:
    logit = float(model["bias"]) + sum(
        float(weight) * float(feature_value)
        for weight, feature_value in zip(model["weights"], standardized_feature_vector)
    )
    probability = stable_sigmoid(logit)
    return {
        "raw_mean": logit,
        "mean": probability,
        "probability": probability,
        "logit": logit,
        "pre_scale": 0.0,
        "head_values": [probability],
    }


def _mlp_predict_value_proxy(model: dict[str, Any], standardized_feature_vector: Sequence[float]) -> dict[str, Any]:
    network = model["network"]
    trunk_output = forward_stack(standardized_feature_vector, network["trunk_layers"])
    head_values = [
        float(forward_stack(trunk_output, head_layers)[0])
        for head_layers in network["head_layers"]
    ]
    raw_mean, pre_scale = mean_and_population_std(head_values)
    probability = clamp_unit_interval(raw_mean)
    return {
        "raw_mean": raw_mean,
        "mean": probability,
        "probability": probability,
        "logit": raw_mean,
        "pre_scale": max(float(model.get("pre_scale_floor", 1e-6)), pre_scale),
        "head_values": head_values,
    }


def load_value_proxy_model(model_path: str | Path) -> dict[str, Any]:
    resolved_path = Path(model_path).resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model_format") not in STAGE0_VALUE_PROXY_SUPPORTED_MODEL_FORMATS:
        raise ValueError(
            f"unsupported Stage-0 value-proxy model format {payload.get('model_format')!r} at {resolved_path}"
        )
    payload["model_path"] = str(resolved_path)
    return payload


def predict_value_proxy(
    model: dict[str, Any],
    *,
    base_feature_vector: Sequence[float],
    proxy_family_id: str | None,
    policy_query_index: int,
) -> dict[str, Any]:
    feature_spec = model["feature_spec"]
    feature_vector = build_value_proxy_feature_vector(
        base_feature_vector,
        proxy_family_id=proxy_family_id,
        policy_query_index=policy_query_index,
        family_ids=feature_spec["family_ids"],
        policy_query_index_scale=float(feature_spec["policy_query_index_scale"]),
        include_base_progress_interaction=bool(feature_spec.get("include_base_progress_interaction", False)),
        include_base_progress_sq_interaction=bool(feature_spec.get("include_base_progress_sq_interaction", False)),
        include_policy_query_index=bool(feature_spec.get("include_policy_query_index", True)),
        include_policy_query_index_sq=bool(feature_spec.get("include_policy_query_index_sq", False)),
        include_family_one_hot=bool(feature_spec.get("include_family_one_hot", True)),
        include_family_progress_interaction=bool(feature_spec.get("include_family_progress_interaction", False)),
        include_family_progress_sq_interaction=bool(feature_spec.get("include_family_progress_sq_interaction", False)),
    )
    standardized_feature_vector = standardize_feature_vector(
        feature_vector,
        mean=model["standardization"]["mean"],
        std=model["standardization"]["std"],
    )
    model_format = str(model.get("model_format") or "")
    if model_format == STAGE0_VALUE_PROXY_MODEL_FORMAT_LEGACY:
        prediction = _legacy_predict_value_proxy(model, standardized_feature_vector)
    elif model_format == STAGE0_VALUE_PROXY_MODEL_FORMAT:
        prediction = _mlp_predict_value_proxy(model, standardized_feature_vector)
    else:
        raise ValueError(f"unsupported Stage-0 value-proxy model format {model_format!r}")

    prediction.update(
        {
            "model_id": str(
                model.get("model_id")
                or (
                    STAGE0_VALUE_PROXY_MODEL_ID
                    if model_format == STAGE0_VALUE_PROXY_MODEL_FORMAT
                    else STAGE0_VALUE_PROXY_MODEL_ID_LEGACY
                )
            ),
            "model_path": str(model.get("model_path") or ""),
            "feature_vector": feature_vector,
        }
    )
    return prediction
