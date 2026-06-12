from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

from stage0_value_proxy import build_value_proxy_feature_vector
from stage0_value_proxy import standardize_feature_vector
from tiny_mlp_artifact import forward_stack


STAGEE_LVD_SELECTOR_MODEL_FORMAT = "stagee_lvd_selector_mlp_v1"
STAGEE_LVD_SELECTOR_MODEL_ID = "stagee_lvd_selector_listwise_mlp_v1"


def load_lvd_selector_model(model_path: str | Path) -> dict[str, Any]:
    resolved_path = Path(model_path).resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model_format") != STAGEE_LVD_SELECTOR_MODEL_FORMAT:
        raise ValueError(f"unsupported LVD selector model format {payload.get('model_format')!r} at {resolved_path}")
    payload["model_path"] = str(resolved_path)
    return payload


def predict_lvd_selector(
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
    network = model["network"]
    trunk_output = forward_stack(standardized_feature_vector, network["trunk_layers"])
    score = float(forward_stack(trunk_output, network["score_head_layers"])[0])
    return {
        "score": score,
        "model_id": str(model.get("model_id") or STAGEE_LVD_SELECTOR_MODEL_ID),
        "model_path": str(model.get("model_path") or ""),
        "feature_vector": feature_vector,
    }
