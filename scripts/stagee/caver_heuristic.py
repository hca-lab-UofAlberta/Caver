from __future__ import annotations

import collections
import functools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from stage0_value_proxy import predict_value_proxy
from stagee_dr_calibration import predict_stagee_dr_calibrator


CAVER_SELECTOR_IMPLEMENTATION_PHASE = "caver_selector_v1"
CAVER_SELECTOR_MODE = "frozen_actionspace_softmax_v1"
CAVER_SELECTOR_MODE_FITTED = "fitted_stage0_value_softmax_v1"
CAVER_SELECTOR_MODE_DR_CALIBRATED = "lagged_dr_calibrated_softmax_v1"
CAVER_ADMISSION_IMPLEMENTATION_PHASE = "caver_admission_v1"
CAVER_ADMISSION_POLICY = "success_lcb_v1"

CAVER_SELECTOR_DEFAULTS: dict[str, float | int] = {
    "value_weight": 2.0,
    "uncertainty_weight": 0.25,
    "diversity_weight": 0.25,
    "novelty_weight": 0.25,
    "provider_value_weight": 0.5,
    "temperature": 0.5,
    "exploration_floor": 0.10,
    "kappa": 0.5,
    "acceptance_threshold": 0.05,
    "novelty_min_history": 32,
    "history_capacity": 1024,
}
PROVIDER_SUMMARY_VECTOR_DIM = 10
PROVIDER_SUMMARY_VERSION = "gesim_future_summary_v1"


def _as_chunk_array(candidate_chunk: Sequence[Sequence[float]]) -> list[list[float]]:
    chunk = [[float(value) for value in step] for step in candidate_chunk]
    if not chunk or not chunk[0]:
        raise ValueError("candidate chunk must be a non-empty 2D sequence")
    width = len(chunk[0])
    for step in chunk:
        if len(step) != width:
            raise ValueError("candidate chunk has inconsistent row widths")
    return chunk


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _stdnorm_chunk(chunk: list[list[float]]) -> list[float]:
    rows = len(chunk)
    cols = len(chunk[0])
    means = [sum(chunk[row][col] for row in range(rows)) / float(rows) for col in range(cols)]
    stds = []
    for col in range(cols):
        variance = sum((chunk[row][col] - means[col]) ** 2 for row in range(rows)) / float(rows)
        std = math.sqrt(variance)
        stds.append(std if std >= 1e-6 else 1.0)
    normalized: list[float] = []
    for row in range(rows):
        for col in range(cols):
            normalized.append((chunk[row][col] - means[col]) / stds[col])
    return normalized


def _mean_pairwise_chunk_distance(candidate_chunks: Sequence[list[list[float]]], *, horizon: int, index: int) -> float:
    if len(candidate_chunks) <= 1:
        return 0.0
    chunk = candidate_chunks[index]
    distances = []
    for other_index, other_chunk in enumerate(candidate_chunks):
        if other_index == index:
            continue
        flat_distance = 0.0
        for row_a, row_b in zip(chunk, other_chunk):
            for value_a, value_b in zip(row_a, row_b):
                flat_distance += (value_a - value_b) ** 2
        distances.append(math.sqrt(flat_distance) / float(horizon))
    return float(sum(distances) / len(distances))


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum - minimum < 1e-9:
        return [0.0 for _ in values]
    scale = maximum - minimum
    return [float((value - minimum) / scale) for value in values]


def _ranknorm(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    ranknorm_values: list[float] = []
    for candidate_index, value in enumerate(values):
        count = 0
        for other_index, other_value in enumerate(values):
            if other_index == candidate_index:
                continue
            if other_value <= value:
                count += 1
        ranknorm_values.append(float(count) / float(len(values) - 1))
    return ranknorm_values


@functools.lru_cache(maxsize=4096)
def _load_provider_summary_path(path_value: str) -> dict[str, Any] | None:
    path = Path(path_value).resolve()
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _zero_provider_summary_vector() -> list[float]:
    return [0.0] * PROVIDER_SUMMARY_VECTOR_DIM


def _extract_provider_summary(candidate_provider_aux: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(candidate_provider_aux, dict):
        return {
            "available": False,
            "summary_vector": _zero_provider_summary_vector(),
            "provider_value_proxy": 0.0,
            "provider_uncertainty_proxy": 0.0,
            "summary_source": None,
            "summary_version": None,
        }

    payload = candidate_provider_aux.get("provider_summary")
    if not isinstance(payload, dict):
        summary_path = candidate_provider_aux.get("provider_summary_path")
        if isinstance(summary_path, str) and summary_path:
            payload = _load_provider_summary_path(summary_path)
    if not isinstance(payload, dict):
        return {
            "available": False,
            "summary_vector": _zero_provider_summary_vector(),
            "provider_value_proxy": 0.0,
            "provider_uncertainty_proxy": 0.0,
            "summary_source": None,
            "summary_version": None,
        }

    raw_vector = payload.get("summary_vector")
    if isinstance(raw_vector, Sequence) and not isinstance(raw_vector, (str, bytes)):
        summary_vector = [float(value) for value in raw_vector[:PROVIDER_SUMMARY_VECTOR_DIM]]
    else:
        summary_vector = []
    if len(summary_vector) < PROVIDER_SUMMARY_VECTOR_DIM:
        summary_vector.extend([0.0] * (PROVIDER_SUMMARY_VECTOR_DIM - len(summary_vector)))

    summary_source = candidate_provider_aux.get("provider_summary_path")
    if summary_source is None and isinstance(payload.get("summary_path"), str):
        summary_source = payload["summary_path"]
    return {
        "available": True,
        "summary_vector": summary_vector,
        "provider_value_proxy": float(payload.get("provider_value_proxy", summary_vector[8])),
        "provider_uncertainty_proxy": float(payload.get("provider_uncertainty_proxy", summary_vector[9])),
        "summary_source": summary_source,
        "summary_version": payload.get("summary_version"),
    }


def compute_candidate_metrics(
    candidate_chunks: Sequence[Sequence[Sequence[float]]],
    candidate_provider_aux: Sequence[dict[str, Any] | None] | None = None,
    history_vectors: Iterable[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    chunk_arrays = [_as_chunk_array(chunk) for chunk in candidate_chunks]
    provider_aux_payloads = list(candidate_provider_aux or [None] * len(chunk_arrays))
    if len(provider_aux_payloads) != len(chunk_arrays):
        raise ValueError("candidate provider payload count does not match candidate count")
    history = [[float(value) for value in history_feature] for history_feature in (history_vectors or [])]
    metrics: list[dict[str, Any]] = []
    raw_values: list[float] = []
    raw_uncertainties: list[float] = []
    raw_diversities: list[float] = []
    raw_novelties: list[float] = []
    base_features: list[list[float]] = []

    for index, chunk in enumerate(chunk_arrays):
        horizon = len(chunk)
        arm_chunk = [step[:-1] if len(step) > 1 else list(step) for step in chunk]
        gripper_values = [step[-1] for step in chunk]
        mean_arm_norm = sum(_vector_norm(step) for step in arm_chunk) / float(horizon)
        mean_full_norm = sum(_vector_norm(step) for step in chunk) / float(horizon)
        smoothness = 0.0
        if horizon > 1:
            diffs = []
            for previous_step, next_step in zip(chunk, chunk[1:]):
                diffs.append(_vector_norm([next_value - prev_value for prev_value, next_value in zip(previous_step, next_step)]))
            smoothness = sum(diffs) / float(len(diffs))
        end_delta = (
            _vector_norm([end_value - start_value for start_value, end_value in zip(chunk[0], chunk[-1])])
            if horizon > 1
            else _vector_norm(chunk[-1])
        )
        gripper_span = max(gripper_values) - min(gripper_values)
        heuristic_raw_value = (0.60 * end_delta) + (0.25 * mean_arm_norm) + (0.15 * gripper_span) - (0.20 * smoothness)
        raw_uncertainty = smoothness
        raw_diversity = _mean_pairwise_chunk_distance(chunk_arrays, horizon=horizon, index=index)
        provider_summary = _extract_provider_summary(provider_aux_payloads[index])
        provider_summary_vector = list(provider_summary["summary_vector"])
        provider_value_proxy = float(provider_summary["provider_value_proxy"])
        provider_uncertainty_proxy = float(provider_summary["provider_uncertainty_proxy"])
        raw_value = heuristic_raw_value + (
            float(CAVER_SELECTOR_DEFAULTS["provider_value_weight"]) * provider_value_proxy
        )

        stdnorm = _stdnorm_chunk(chunk)
        base_feature = [raw_value, *stdnorm, raw_diversity, raw_uncertainty]
        base_feature.extend(provider_summary_vector)
        if len(history) < int(CAVER_SELECTOR_DEFAULTS["novelty_min_history"]):
            raw_novelty = 0.0
        else:
            raw_novelty = min(
                _vector_norm(
                    [
                        feature_value - history_value
                        for feature_value, history_value in zip(base_feature, history_feature)
                    ]
                )
                / float(len(base_feature))
                for history_feature in history
            )

        metrics.append(
            {
                "chunk_action_horizon": horizon,
                "action_summary_dim": len(base_feature),
                "mean_arm_norm": mean_arm_norm,
                "mean_full_norm": mean_full_norm,
                "smoothness": smoothness,
                "end_delta": end_delta,
                "gripper_span": gripper_span,
                "gripper_final": gripper_values[-1],
                "heuristic_raw_value_proxy": float(heuristic_raw_value),
                "provider_summary_available": bool(provider_summary["available"]),
                "provider_summary_version": provider_summary["summary_version"],
                "provider_summary_source": provider_summary["summary_source"],
                "provider_summary_dim": PROVIDER_SUMMARY_VECTOR_DIM,
                "provider_summary_vector": list(provider_summary_vector),
                "provider_value_proxy": provider_value_proxy,
                "provider_uncertainty_proxy": provider_uncertainty_proxy,
            }
        )
        raw_values.append(raw_value)
        raw_uncertainties.append(raw_uncertainty)
        raw_diversities.append(raw_diversity)
        raw_novelties.append(raw_novelty)
        base_features.append(base_feature)

    value_ranknorm = _ranknorm(raw_values)
    normalized_uncertainty = _minmax(raw_uncertainties)
    normalized_diversity = _minmax(raw_diversities)
    normalized_novelty = _minmax(raw_novelties)

    for index, metric in enumerate(metrics):
        metric.update(
            {
                "raw_value_proxy": float(raw_values[index]),
                "value_ranknorm": float(value_ranknorm[index]),
                "raw_uncertainty_proxy": float(raw_uncertainties[index]),
                "uncertainty_normalized": float(normalized_uncertainty[index]),
                "raw_diversity_proxy": float(raw_diversities[index]),
                "diversity_normalized": float(normalized_diversity[index]),
                "raw_novelty_proxy": float(raw_novelties[index]),
                "novelty_normalized": float(normalized_novelty[index]),
                "base_feature_vector": list(base_features[index]),
            }
        )
    return metrics


def compute_selector_decision(
    candidate_chunks: Sequence[Sequence[Sequence[float]]],
    *,
    safe_candidate_mask: Sequence[bool] | None,
    candidate_provider_aux: Sequence[dict[str, Any] | None] | None,
    history_vectors: Iterable[Sequence[float]] | None,
    rng: Any | None,
    value_proxy_model: dict[str, Any] | None = None,
    dr_calibrator_model: dict[str, Any] | None = None,
    proxy_family_id: str | None = None,
    policy_query_index: int = 0,
) -> dict[str, Any]:
    history = [[float(value) for value in history_feature] for history_feature in (history_vectors or [])]
    metrics = compute_candidate_metrics(
        candidate_chunks,
        candidate_provider_aux=candidate_provider_aux,
        history_vectors=history,
    )
    safe_mask = list(safe_candidate_mask or [True] * len(metrics))
    if len(safe_mask) != len(metrics):
        raise ValueError("safe candidate mask length does not match candidate count")
    safe_indices = [index for index, is_safe in enumerate(safe_mask) if is_safe]
    if not safe_indices:
        raise ValueError("selector requires at least one safety-approved candidate")

    value_proxy_source = "heuristic_actionspace_v1"
    value_proxy_model_id: str | None = None
    utility_scale_source = "heuristic_uncertainty_v1"
    utility_scale_model_id: str | None = None
    selector_mode = CAVER_SELECTOR_MODE
    if dr_calibrator_model is not None:
        selector_mode = CAVER_SELECTOR_MODE_DR_CALIBRATED
        value_proxy_source = str(dr_calibrator_model.get("model_id") or CAVER_SELECTOR_MODE_DR_CALIBRATED)
        value_proxy_model_id = value_proxy_source
        utility_scale_source = value_proxy_source
        utility_scale_model_id = value_proxy_source
        value_inputs = []
        scale_inputs = []
        for metric in metrics:
            prediction = predict_stagee_dr_calibrator(
                dr_calibrator_model,
                base_feature_vector=metric["base_feature_vector"],
                proxy_family_id=proxy_family_id,
                policy_query_index=policy_query_index,
            )
            corrected_mean = float(prediction["mean"])
            corrected_scale = float(prediction["scale"])
            value_inputs.append(corrected_mean)
            scale_inputs.append(corrected_scale)
            metric["heuristic_raw_value_proxy"] = float(metric["raw_value_proxy"])
            metric["heuristic_value_ranknorm"] = float(metric["value_ranknorm"])
            metric["lagged_nuisance_mean"] = corrected_mean
            metric["lagged_dr_utility_mean"] = corrected_mean
            metric["lagged_dr_utility_raw_mean"] = float(prediction["raw_mean"])
            metric["lagged_dr_utility_scale"] = corrected_scale
            metric["admission_value_proxy"] = corrected_mean
            metric["admission_uncertainty_proxy"] = corrected_scale
            metric["value_proxy_source"] = value_proxy_source
            metric["value_proxy_model_path"] = str(prediction["model_path"])
            metric["utility_scale_source"] = utility_scale_source
        value_ranknorm = _ranknorm(value_inputs)
        normalized_uncertainty = _minmax(scale_inputs)
        for index, metric in enumerate(metrics):
            metric["raw_value_proxy"] = float(value_inputs[index])
            metric["value_ranknorm"] = float(value_ranknorm[index])
            metric["raw_uncertainty_proxy"] = float(scale_inputs[index])
            metric["uncertainty_normalized"] = float(normalized_uncertainty[index])
    elif value_proxy_model is None:
        value_inputs = [float(metric["raw_value_proxy"]) for metric in metrics]
        value_ranknorm = _ranknorm(value_inputs)
        for index, metric in enumerate(metrics):
            metric["heuristic_raw_value_proxy"] = float(metric["raw_value_proxy"])
            metric["heuristic_value_ranknorm"] = float(metric["value_ranknorm"])
            metric["lagged_nuisance_mean"] = float(value_inputs[index])
            metric["admission_value_proxy"] = float(value_ranknorm[index])
            metric["value_proxy_source"] = value_proxy_source
    else:
        selector_mode = CAVER_SELECTOR_MODE_FITTED
        value_proxy_source = str(value_proxy_model.get("model_id") or CAVER_SELECTOR_MODE_FITTED)
        value_proxy_model_id = value_proxy_source
        utility_scale_source = value_proxy_source
        utility_scale_model_id = value_proxy_source
        value_inputs = []
        scale_inputs = []
        for metric in metrics:
            prediction = predict_value_proxy(
                value_proxy_model,
                base_feature_vector=metric["base_feature_vector"],
                proxy_family_id=proxy_family_id,
                policy_query_index=policy_query_index,
            )
            predicted_mean = float(prediction.get("mean", prediction["probability"]))
            predicted_scale = float(prediction.get("pre_scale", metric["raw_uncertainty_proxy"]))
            value_inputs.append(predicted_mean)
            scale_inputs.append(predicted_scale)
            metric["heuristic_raw_value_proxy"] = float(metric["raw_value_proxy"])
            metric["heuristic_value_ranknorm"] = float(metric["value_ranknorm"])
            metric["lagged_nuisance_mean"] = predicted_mean
            metric["admission_value_proxy"] = predicted_mean
            metric["value_proxy_logit"] = float(prediction["logit"])
            metric["value_proxy_pre_scale"] = predicted_scale
            metric["value_proxy_source"] = value_proxy_source
            metric["value_proxy_model_path"] = str(prediction["model_path"])
            metric["utility_scale_source"] = utility_scale_source
        value_ranknorm = _ranknorm(value_inputs)
        normalized_uncertainty = _minmax(scale_inputs)
        for index, metric in enumerate(metrics):
            metric["raw_value_proxy"] = float(value_inputs[index])
            metric["value_ranknorm"] = float(value_ranknorm[index])
            metric["raw_uncertainty_proxy"] = float(scale_inputs[index])
            metric["uncertainty_normalized"] = float(normalized_uncertainty[index])

    raw_scores = []
    for metric in metrics:
        raw_scores.append(
            (float(CAVER_SELECTOR_DEFAULTS["value_weight"]) * metric["value_ranknorm"])
            + (float(CAVER_SELECTOR_DEFAULTS["uncertainty_weight"]) * metric["uncertainty_normalized"])
            + (float(CAVER_SELECTOR_DEFAULTS["diversity_weight"]) * metric["diversity_normalized"])
            + (float(CAVER_SELECTOR_DEFAULTS["novelty_weight"]) * metric["novelty_normalized"])
        )

    safe_scores = [raw_scores[index] for index in safe_indices]
    temperature = float(CAVER_SELECTOR_DEFAULTS["temperature"])
    maximum_logit = max(safe_scores) / temperature
    safe_softmax = [math.exp((score / temperature) - maximum_logit) for score in safe_scores]
    softmax_total = sum(safe_softmax)
    safe_softmax = [value / softmax_total for value in safe_softmax]

    candidate_count = len(metrics)
    exploration_floor = float(CAVER_SELECTOR_DEFAULTS["exploration_floor"])
    candidate_probabilities = [0.0] * candidate_count
    for local_index, candidate_index in enumerate(safe_indices):
        candidate_probabilities[candidate_index] = float((1.0 - exploration_floor) * safe_softmax[local_index])
    uniform_mass = exploration_floor / float(candidate_count)
    candidate_probabilities = [
        probability + (uniform_mass if safe_mask[index] else 0.0) for index, probability in enumerate(candidate_probabilities)
    ]
    safe_total = sum(candidate_probabilities[index] for index in safe_indices)
    candidate_probabilities = [
        float(probability / safe_total) if safe_mask[index] else 0.0
        for index, probability in enumerate(candidate_probabilities)
    ]

    if rng is None:
        selected_candidate_index = max(safe_indices, key=lambda index: candidate_probabilities[index])
    else:
        safe_probability_vector = [candidate_probabilities[index] for index in safe_indices]
        safe_probability_total = sum(safe_probability_vector)
        safe_probability_vector = [value / safe_probability_total for value in safe_probability_vector]
        draw = float(rng.random())
        cumulative = 0.0
        selected_candidate_index = safe_indices[-1]
        for candidate_index, probability in zip(safe_indices, safe_probability_vector):
            cumulative += probability
            if draw <= cumulative:
                selected_candidate_index = candidate_index
                break

    selected_metrics = metrics[selected_candidate_index]
    return {
        "selector_mode": selector_mode,
        "implementation_phase": CAVER_SELECTOR_IMPLEMENTATION_PHASE,
        "candidate_metrics": metrics,
        "candidate_scores": [float(score) for score in raw_scores],
        "candidate_probabilities": [float(probability) for probability in candidate_probabilities],
        "safe_candidate_indices": safe_indices,
        "safe_candidate_count": len(safe_indices),
        "selected_candidate_index": selected_candidate_index,
        "selected_candidate_probability": float(candidate_probabilities[selected_candidate_index]),
        "selected_base_feature_vector": selected_metrics["base_feature_vector"],
        "history_size": len(history),
        "selector_temperature": temperature,
        "selector_exploration_floor": exploration_floor,
        "value_proxy_source": value_proxy_source,
        "value_proxy_model_id": value_proxy_model_id,
        "utility_scale_source": utility_scale_source,
        "utility_scale_model_id": utility_scale_model_id,
        "selector_weights": {
            "value_weight": float(CAVER_SELECTOR_DEFAULTS["value_weight"]),
            "uncertainty_weight": float(CAVER_SELECTOR_DEFAULTS["uncertainty_weight"]),
            "diversity_weight": float(CAVER_SELECTOR_DEFAULTS["diversity_weight"]),
            "novelty_weight": float(CAVER_SELECTOR_DEFAULTS["novelty_weight"]),
            "provider_value_weight": float(CAVER_SELECTOR_DEFAULTS["provider_value_weight"]),
        },
    }


def make_selector_history() -> collections.deque[list[float]]:
    return collections.deque(maxlen=int(CAVER_SELECTOR_DEFAULTS["history_capacity"]))


def append_selector_history(history: collections.deque[list[float]], feature_vector: Sequence[float]) -> None:
    history.append([float(value) for value in feature_vector])


def selected_metric_from_record(trace_record: dict[str, Any]) -> dict[str, Any]:
    selector = trace_record.get("selector", {})
    selected_index = int(selector["selected_candidate_index"])
    if "candidate_metric_table" in selector:
        metric_table = selector["candidate_metric_table"]
        return dict(metric_table[selected_index])
    candidate_chunks = trace_record.get("candidate_chunks")
    if candidate_chunks is None:
        raise ValueError(f"trace record for context {trace_record.get('context_id')} is missing candidate chunks")
    metrics = compute_candidate_metrics(
        candidate_chunks,
        candidate_provider_aux=trace_record.get("candidate_provider_aux"),
        history_vectors=None,
    )
    metric = dict(metrics[selected_index])
    metric.update(
        {
            "selected_candidate_probability": selector.get("selected_candidate_probability"),
            "selected_candidate_index": selected_index,
        }
    )
    return metric


def summarize_admission_context(
    *,
    context: dict[str, Any],
    context_trace_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected_metrics = [selected_metric_from_record(record) for record in context_trace_records]
    return summarize_admission_metrics(context=context, selected_metrics=selected_metrics)


def summarize_admission_metrics(
    *,
    context: dict[str, Any],
    selected_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if selected_metrics:
        utility_mean = sum(
            float(metric.get("admission_value_proxy", metric["value_ranknorm"])) for metric in selected_metrics
        ) / float(len(selected_metrics))
        uncertainty_mean = sum(
            float(
                metric.get(
                    "admission_uncertainty_proxy",
                    metric.get("raw_uncertainty_proxy", metric["uncertainty_normalized"]),
                )
            )
            for metric in selected_metrics
        ) / float(len(selected_metrics))
        diversity_mean = sum(metric["diversity_normalized"] for metric in selected_metrics) / float(
            len(selected_metrics)
        )
        novelty_mean = sum(metric["novelty_normalized"] for metric in selected_metrics) / float(len(selected_metrics))
        lcb = utility_mean - (float(CAVER_SELECTOR_DEFAULTS["kappa"]) * uncertainty_mean)
    else:
        utility_mean = 0.0
        uncertainty_mean = 0.0
        diversity_mean = 0.0
        novelty_mean = 0.0
        lcb = 0.0

    success = bool(context.get("success"))
    has_error = context.get("error") is not None
    safety_abort = bool(context.get("budget", {}).get("safety_abort", False))
    has_trace = bool(selected_metrics)
    admit_for_training = has_trace and not has_error and not safety_abort and success and (
        lcb > float(CAVER_SELECTOR_DEFAULTS["acceptance_threshold"])
    )

    if not has_trace:
        admission_reason = "missing_trace_records"
    elif has_error:
        admission_reason = "context_error"
    elif safety_abort:
        admission_reason = "safety_abort"
    elif not success:
        admission_reason = "failed_execution"
    elif lcb <= 0.0:
        admission_reason = "success_nonpositive_lcb"
    elif lcb <= float(CAVER_SELECTOR_DEFAULTS["acceptance_threshold"]):
        admission_reason = "success_abstain_low_confidence"
    else:
        admission_reason = "success_high_confidence"

    return {
        "selected_query_count": len(selected_metrics),
        "executed_value_mean": utility_mean,
        "executed_uncertainty_mean": uncertainty_mean,
        "executed_diversity_mean": diversity_mean,
        "executed_novelty_mean": novelty_mean,
        "executed_lcb": lcb,
        "admit_for_training": admit_for_training,
        "admission_reason": admission_reason,
        "admission_confidence": max(0.0, min(1.0, lcb)),
        "acceptance_threshold": float(CAVER_SELECTOR_DEFAULTS["acceptance_threshold"]),
        "kappa": float(CAVER_SELECTOR_DEFAULTS["kappa"]),
    }
