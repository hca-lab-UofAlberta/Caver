from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

from stage0_value_proxy import build_value_proxy_feature_vector
from stage0_value_proxy import standardize_feature_vector
from tiny_mlp_artifact import bounded_sigmoid
from tiny_mlp_artifact import clamp_unit_interval
from tiny_mlp_artifact import forward_stack


STAGEE_DR_DATASET_FORMAT = "stagee_dr_candidate_dataset_v1"
STAGEE_DR_CALIBRATOR_MODEL_FORMAT_LEGACY = "stagee_dr_calibrator_v1"
STAGEE_DR_CALIBRATOR_MODEL_FORMAT = "stagee_dr_calibrator_mlp_v2"
STAGEE_DR_CALIBRATOR_SUPPORTED_MODEL_FORMATS = {
    STAGEE_DR_CALIBRATOR_MODEL_FORMAT_LEGACY,
    STAGEE_DR_CALIBRATOR_MODEL_FORMAT,
}
STAGEE_DR_CALIBRATOR_MODEL_ID_LEGACY = "stagee_dr_calibrator_ridge_v1"
STAGEE_DR_CALIBRATOR_MODEL_ID = "stagee_dr_calibrator_mlp_width256_v2"


def dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(float(weight) * float(feature) for weight, feature in zip(weights, features))


def load_stagee_dr_calibrator_model(model_path: str | Path) -> dict[str, Any]:
    resolved_path = Path(model_path).resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model_format") not in STAGEE_DR_CALIBRATOR_SUPPORTED_MODEL_FORMATS:
        raise ValueError(
            f"unsupported Stage-E DR calibrator format {payload.get('model_format')!r} at {resolved_path}"
        )
    payload["model_path"] = str(resolved_path)
    return payload


def _legacy_predict_stagee_dr_calibrator(
    model: dict[str, Any],
    standardized_feature_vector: Sequence[float],
    *,
    proxy_family_id: str | None,
) -> dict[str, Any]:
    raw_mean = float(model["bias"]) + dot(model["weights"], standardized_feature_vector)
    corrected_mean = clamp_unit_interval(raw_mean)
    residual_scale_by_family = model.get("residual_scale_by_family", {})
    raw_scale = residual_scale_by_family.get(str(proxy_family_id), model["residual_scale_global"])
    corrected_scale = max(float(model.get("residual_scale_floor", 1e-3)), float(raw_scale))
    return {
        "raw_mean": raw_mean,
        "mean": corrected_mean,
        "scale": corrected_scale,
        "raw_scale": float(raw_scale),
    }


def _mlp_predict_stagee_dr_calibrator(model: dict[str, Any], standardized_feature_vector: Sequence[float]) -> dict[str, Any]:
    network = model["network"]
    trunk_output = forward_stack(standardized_feature_vector, network["trunk_layers"])
    raw_mean = float(forward_stack(trunk_output, network["mean_head_layers"])[0])
    raw_scale = float(forward_stack(trunk_output, network["scale_head_layers"])[0])
    scale_bounds = model.get("scale_bounds", {})
    floor = float(scale_bounds.get("floor", 0.02))
    ceiling = float(scale_bounds.get("ceiling", 0.5))
    corrected_scale = bounded_sigmoid(raw_scale, lower=floor, upper=ceiling)
    corrected_mean = raw_mean
    return {
        "raw_mean": raw_mean,
        "mean": corrected_mean,
        "clipped_mean": clamp_unit_interval(raw_mean),
        "scale": corrected_scale,
        "raw_scale": raw_scale,
    }


def predict_stagee_dr_calibrator(
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
    if model_format == STAGEE_DR_CALIBRATOR_MODEL_FORMAT_LEGACY:
        prediction = _legacy_predict_stagee_dr_calibrator(
            model,
            standardized_feature_vector,
            proxy_family_id=proxy_family_id,
        )
    elif model_format == STAGEE_DR_CALIBRATOR_MODEL_FORMAT:
        prediction = _mlp_predict_stagee_dr_calibrator(model, standardized_feature_vector)
    else:
        raise ValueError(f"unsupported Stage-E DR calibrator format {model_format!r}")
    prediction.update(
        {
            "model_id": str(
                model.get("model_id")
                or (
                    STAGEE_DR_CALIBRATOR_MODEL_ID
                    if model_format == STAGEE_DR_CALIBRATOR_MODEL_FORMAT
                    else STAGEE_DR_CALIBRATOR_MODEL_ID_LEGACY
                )
            ),
            "model_path": str(model.get("model_path") or ""),
            "feature_vector": feature_vector,
        }
    )
    return prediction
