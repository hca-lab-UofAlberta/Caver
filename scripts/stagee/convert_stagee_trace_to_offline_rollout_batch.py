#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from openpi.models import tokenizer as openpi_tokenizer

TRACE_INPUT_FORMAT = "stagee_trace_source_manifest_v1"
ROLL_OUT_BATCH_FORMAT = "stagee_exact_offline_rollout_batch_v1"


@dataclass
class ChunkPayload:
    context_id: str
    policy_query_index: int
    proxy_family_id: str | None
    partition_name: str | None
    task_id: int | None
    prompt: str
    primitive_steps: int
    reward_sum: float
    done: bool
    termination: bool
    truncation: bool
    success: bool
    observation_image: np.ndarray
    observation_state: np.ndarray
    observation_wrist_image: np.ndarray | None
    tokenized_prompt: np.ndarray
    tokenized_prompt_mask: np.ndarray
    prev_logprobs: np.ndarray | None
    nft_xt: np.ndarray
    nft_v: np.ndarray
    nft_xnext: np.ndarray
    nft_step_index: np.ndarray
    nft_noise_level: np.ndarray


@dataclass
class ContextPayload:
    context_id: str
    proxy_family_id: str | None
    partition_name: str | None
    task_id: int | None
    task_name: str | None
    prompt: str
    chunks: list[ChunkPayload] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert admitted Stage-E exact traces into an RLinf offline rollout batch "
            "for NFT/StepNFT-style post-train updates."
        )
    )
    parser.add_argument(
        "--trace-path",
        required=True,
        help=(
            "Admitted-trace JSONL path or trace-source manifest emitted by the Stage-E "
            "artifact builder."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output .pt path for the offline rollout batch payload.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON summary path describing the converted rollout batch.",
    )
    parser.add_argument(
        "--openpi-config-name",
        default="pi05_libero",
        help="OpenPI config name used to recover tokenizer settings when prompt tokens are missing.",
    )
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=None,
        help="Optional tokenizer max length override.",
    )
    parser.add_argument(
        "--discrete-state-input",
        action="store_true",
        help="Retokenize prompts with Pi0.5-style discretized state text when needed.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of trace records read.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Optional cap on the number of contexts converted.",
    )
    parser.add_argument(
        "--include-prev-logprobs",
        action="store_true",
        help="Preserve prev_logprobs when every converted chunk provides them.",
    )
    return parser.parse_args()


def resolve_tokenizer_settings(args: argparse.Namespace) -> tuple[int, bool]:
    discrete_state_input = args.discrete_state_input
    max_token_len = args.max_token_len

    if max_token_len is not None:
        return max_token_len, discrete_state_input

    try:
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        openpi_config = get_openpi_config(args.openpi_config_name)
        max_token_len = int(openpi_config.model.max_token_len)
        if not args.discrete_state_input:
            discrete_state_input = bool(
                getattr(openpi_config.model, "discrete_state_input", False)
            )
        return max_token_len, discrete_state_input
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "failed to resolve tokenizer settings from config %s: %s; using heuristic defaults",
            args.openpi_config_name,
            exc,
        )
        return (200 if "pi05" in args.openpi_config_name else 48), discrete_state_input


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_trace_source_manifest(trace_path: Path) -> dict[str, Any] | None:
    first_nonempty: str | None = None
    with trace_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            first_nonempty = line
            break
    if first_nonempty is None:
        return None
    try:
        payload = json.loads(first_nonempty)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("trace_input_format") == TRACE_INPUT_FORMAT:
        return payload
    return None


def iter_trace_records_from_source(
    trace_path: Path,
    *,
    completed_prefix_contexts: int | None = None,
) -> Iterator[dict[str, Any]]:
    contexts_seen: set[str] = set()
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if completed_prefix_contexts is not None and len(contexts_seen) >= completed_prefix_contexts:
                    break
                continue
            context_id = str(record["context_id"])
            if completed_prefix_contexts is not None and context_id not in contexts_seen:
                if len(contexts_seen) >= completed_prefix_contexts:
                    break
                contexts_seen.add(context_id)
            yield record


def iter_trace_records(trace_path: Path, max_records: int | None) -> Iterator[dict[str, Any]]:
    manifest = load_trace_source_manifest(trace_path)
    produced = 0
    if manifest is None:
        yield from _iter_jsonl_trace_records(trace_path, max_records)
        return
    for source in manifest.get("sources", []):
        source_path = Path(source["path"]).resolve()
        completed_prefix_contexts = source.get("completed_prefix_contexts")
        for record in iter_trace_records_from_source(
            source_path,
            completed_prefix_contexts=(
                None if completed_prefix_contexts is None else int(completed_prefix_contexts)
            ),
        ):
            yield record
            produced += 1
            if max_records is not None and produced >= max_records:
                return
    if produced == 0:
        raise ValueError(f"trace manifest produced zero records: {trace_path}")


def _iter_jsonl_trace_records(
    trace_path: Path,
    max_records: int | None,
) -> Iterator[dict[str, Any]]:
    produced = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"failed to decode trace line {line_number} from {trace_path}"
                ) from exc
            produced += 1
            if max_records is not None and produced >= max_records:
                return
    if produced == 0:
        raise ValueError(f"trace path produced zero records: {trace_path}")


def tokenize_prompt(
    tokenizer: openpi_tokenizer.PaligemmaTokenizer,
    prompt: str,
    state: np.ndarray,
    *,
    discrete_state_input: bool,
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not discrete_state_input and prompt in token_cache:
        return token_cache[prompt]

    token_state = state if discrete_state_input else None
    tokens, token_mask = tokenizer.tokenize(prompt, token_state)
    tokens = np.asarray(tokens, dtype=np.int32)
    token_mask = np.asarray(token_mask, dtype=np.bool_)
    if not discrete_state_input:
        token_cache[prompt] = (tokens, token_mask)
    return tokens, token_mask


def coerce_array(name: str, value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return array.reshape(())
    return np.ascontiguousarray(array)


def coerce_optional_array(
    name: str,
    value: Any | None,
    *,
    dtype: np.dtype | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    return coerce_array(name, value, dtype=dtype)


def validate_record_lengths(record: dict[str, Any]) -> None:
    steps_executed = len(record["actions"])
    expected_lengths = {
        "rewards": len(record["rewards"]),
        "dones": len(record["dones"]),
        "terminations": len(record["terminations"]),
        "truncations": len(record["truncations"]),
        "success_once": len(record["success_once"]),
    }
    bad_lengths = {key: value for key, value in expected_lengths.items() if value != steps_executed}
    if bad_lengths:
        raise ValueError(
            f"trace record {record['context_id']} policy_query_index={record['policy_query_index']} "
            f"has inconsistent lengths: actions={steps_executed}, {bad_lengths}"
        )


def _normalize_task_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def align_exact_chunk_horizon(
    *,
    context_id: str,
    policy_query_index: int,
    nft_xt: np.ndarray,
    nft_v: np.ndarray,
    nft_xnext: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nft_xt.ndim < 2 or nft_v.ndim < 2 or nft_xnext.ndim < 2:
        raise ValueError(
            f"context {context_id} policy_query_index={policy_query_index} has malformed exact tensors: "
            f"nft_xt={tuple(nft_xt.shape)} nft_v={tuple(nft_v.shape)} nft_xnext={tuple(nft_xnext.shape)}"
        )
    chunk_horizon = int(nft_v.shape[0])
    if chunk_horizon <= 0:
        raise ValueError(
            f"context {context_id} policy_query_index={policy_query_index} has empty nft_v tensor"
        )
    if nft_xt.shape[0] < chunk_horizon or nft_xnext.shape[0] < chunk_horizon:
        raise ValueError(
            f"context {context_id} policy_query_index={policy_query_index} has insufficient exact horizon: "
            f"nft_xt={tuple(nft_xt.shape)} nft_v={tuple(nft_v.shape)} nft_xnext={tuple(nft_xnext.shape)}"
        )
    return (
        np.ascontiguousarray(nft_xt),
        np.ascontiguousarray(nft_v),
        np.ascontiguousarray(nft_xnext),
    )


def extract_chunk_payload(
    record: dict[str, Any],
    *,
    tokenizer: openpi_tokenizer.PaligemmaTokenizer,
    discrete_state_input: bool,
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> ChunkPayload:
    validate_record_lengths(record)
    selected_policy_aux = record.get("selected_policy_aux")
    if not isinstance(selected_policy_aux, dict):
        raise ValueError(
            f"context {record.get('context_id')} policy_query_index={record.get('policy_query_index')} "
            "is missing selected_policy_aux; exact rollout conversion requires exact-payload traces"
        )
    forward_inputs = selected_policy_aux.get("forward_inputs")
    if not isinstance(forward_inputs, dict):
        raise ValueError(
            f"context {record.get('context_id')} policy_query_index={record.get('policy_query_index')} "
            "is missing selected_policy_aux.forward_inputs"
        )

    missing_exact = [
        key
        for key in ("nft_xt", "nft_v", "nft_xnext", "nft_step_index", "nft_noise_level")
        if key not in forward_inputs
    ]
    if missing_exact:
        raise ValueError(
            f"context {record.get('context_id')} policy_query_index={record.get('policy_query_index')} "
            f"is missing exact rollout keys {missing_exact}"
        )

    prompt = str(record["prompt"])
    obs_payload = record["obs"]
    state = coerce_array(
        "observation/state",
        forward_inputs.get("observation/state", obs_payload["state"]),
        dtype=np.float32,
    )
    tokenized_prompt = coerce_optional_array(
        "tokenized_prompt", forward_inputs.get("tokenized_prompt"), dtype=np.int32
    )
    tokenized_prompt_mask = coerce_optional_array(
        "tokenized_prompt_mask",
        forward_inputs.get("tokenized_prompt_mask"),
        dtype=np.bool_,
    )
    if tokenized_prompt is None or tokenized_prompt_mask is None:
        tokenized_prompt, tokenized_prompt_mask = tokenize_prompt(
            tokenizer,
            prompt,
            state,
            discrete_state_input=discrete_state_input,
            token_cache=token_cache,
        )

    rewards = np.asarray(record["rewards"], dtype=np.float32)
    dones = np.asarray(record["dones"], dtype=np.bool_)
    terminations = np.asarray(record["terminations"], dtype=np.bool_)
    truncations = np.asarray(record["truncations"], dtype=np.bool_)
    success_once = np.asarray(record["success_once"], dtype=np.bool_)

    nft_xt = coerce_array("nft_xt", forward_inputs["nft_xt"], dtype=np.float32)
    nft_v = coerce_array("nft_v", forward_inputs["nft_v"], dtype=np.float32)
    nft_xnext = coerce_array("nft_xnext", forward_inputs["nft_xnext"], dtype=np.float32)
    nft_xt, nft_v, nft_xnext = align_exact_chunk_horizon(
        context_id=str(record["context_id"]),
        policy_query_index=int(record["policy_query_index"]),
        nft_xt=nft_xt,
        nft_v=nft_v,
        nft_xnext=nft_xnext,
    )

    return ChunkPayload(
        context_id=str(record["context_id"]),
        policy_query_index=int(record["policy_query_index"]),
        proxy_family_id=(
            None if record.get("proxy_family_id") is None else str(record["proxy_family_id"])
        ),
        partition_name=(
            None if record.get("partition_name") is None else str(record["partition_name"])
        ),
        task_id=_normalize_task_id(record.get("task_id")),
        prompt=prompt,
        primitive_steps=len(record["actions"]),
        reward_sum=float(rewards.sum()),
        done=bool(dones.any()),
        termination=bool(terminations.any()),
        truncation=bool(truncations.any()),
        success=bool(success_once.any()),
        observation_image=coerce_array(
            "observation/image",
            forward_inputs.get("observation/image", obs_payload["image"]),
            dtype=np.uint8,
        ),
        observation_state=state,
        observation_wrist_image=coerce_optional_array(
            "observation/wrist_image",
            forward_inputs.get("observation/wrist_image", obs_payload.get("wrist_image")),
            dtype=np.uint8,
        ),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        prev_logprobs=coerce_optional_array(
            "prev_logprobs",
            selected_policy_aux.get("prev_logprobs"),
            dtype=np.float32,
        ),
        nft_xt=nft_xt,
        nft_v=nft_v,
        nft_xnext=nft_xnext,
        nft_step_index=coerce_array(
            "nft_step_index", forward_inputs["nft_step_index"], dtype=np.int64
        ),
        nft_noise_level=coerce_array(
            "nft_noise_level", forward_inputs["nft_noise_level"], dtype=np.float32
        ),
    )


def ensure_same_shape(name: str, current: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if tuple(current.shape) != expected_shape:
        raise ValueError(
            f"inconsistent {name} shape {tuple(current.shape)}; expected {expected_shape}"
        )


def build_rollout_payload(
    contexts: list[ContextPayload],
    *,
    include_prev_logprobs: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not contexts:
        raise ValueError("conversion produced zero contexts")
    first_chunk = next(
        (chunk for context in contexts for chunk in context.chunks),
        None,
    )
    if first_chunk is None:
        raise ValueError("conversion produced zero chunk records")

    num_chunk = max(len(context.chunks) for context in contexts)
    batch_size = len(contexts)

    image_shape = tuple(first_chunk.observation_image.shape)
    state_shape = tuple(first_chunk.observation_state.shape)
    token_shape = tuple(first_chunk.tokenized_prompt.shape)
    token_mask_shape = tuple(first_chunk.tokenized_prompt_mask.shape)
    nft_xt_shape = tuple(first_chunk.nft_xt.shape)
    nft_v_shape = tuple(first_chunk.nft_v.shape)
    nft_xnext_shape = tuple(first_chunk.nft_xnext.shape)
    nft_step_index_shape = tuple(first_chunk.nft_step_index.shape)
    nft_noise_level_shape = tuple(first_chunk.nft_noise_level.shape)
    prev_logprob_shape = (
        None if first_chunk.prev_logprobs is None else tuple(first_chunk.prev_logprobs.shape)
    )
    has_wrist_image = any(chunk.observation_wrist_image is not None for context in contexts for chunk in context.chunks)
    wrist_shape = None
    if has_wrist_image:
        wrist_chunk = next(
            chunk
            for context in contexts
            for chunk in context.chunks
            if chunk.observation_wrist_image is not None
        )
        wrist_shape = tuple(wrist_chunk.observation_wrist_image.shape)

    rewards = torch.zeros((num_chunk, batch_size, 1), dtype=torch.float32)
    success_once = torch.zeros((num_chunk, batch_size, 1), dtype=torch.bool)
    dones = torch.ones((num_chunk + 1, batch_size, 1), dtype=torch.bool)
    dones[0] = False
    terminations = torch.zeros((num_chunk + 1, batch_size, 1), dtype=torch.bool)
    truncations = torch.zeros((num_chunk + 1, batch_size, 1), dtype=torch.bool)
    loss_mask = torch.zeros((num_chunk, batch_size, 1), dtype=torch.bool)

    observation_image = torch.zeros(
        (num_chunk, batch_size, *image_shape),
        dtype=torch.uint8,
    )
    observation_state = torch.zeros(
        (num_chunk, batch_size, *state_shape),
        dtype=torch.float32,
    )
    tokenized_prompt = torch.zeros(
        (num_chunk, batch_size, *token_shape),
        dtype=torch.int32,
    )
    tokenized_prompt_mask = torch.zeros(
        (num_chunk, batch_size, *token_mask_shape),
        dtype=torch.bool,
    )
    observation_wrist_image = None
    if has_wrist_image:
        observation_wrist_image = torch.zeros(
            (num_chunk, batch_size, *wrist_shape),
            dtype=torch.uint8,
        )

    nft_xt = torch.zeros((num_chunk, batch_size, *nft_xt_shape), dtype=torch.float32)
    nft_v = torch.zeros((num_chunk, batch_size, *nft_v_shape), dtype=torch.float32)
    nft_xnext = torch.zeros((num_chunk, batch_size, *nft_xnext_shape), dtype=torch.float32)
    nft_step_index = torch.zeros((num_chunk, batch_size, *nft_step_index_shape), dtype=torch.int64)
    nft_noise_level = torch.zeros(
        (num_chunk, batch_size, *nft_noise_level_shape),
        dtype=torch.float32,
    )

    prev_logprobs = None
    if include_prev_logprobs and prev_logprob_shape is not None:
        prev_logprobs = torch.zeros(
            (num_chunk, batch_size, *prev_logprob_shape),
            dtype=torch.float32,
        )

    primitive_steps_total = 0
    family_counts: collections.Counter[str] = collections.Counter()
    partition_counts: collections.Counter[str] = collections.Counter()
    context_chunk_counts: dict[str, int] = {}
    context_task_names: dict[str, str | None] = {}

    for batch_index, context in enumerate(contexts):
        sorted_chunks = sorted(context.chunks, key=lambda item: item.policy_query_index)
        context_chunk_counts[context.context_id] = len(sorted_chunks)
        context_task_names[context.context_id] = context.task_name
        if context.proxy_family_id is not None:
            family_counts[context.proxy_family_id] += 1
        if context.partition_name is not None:
            partition_counts[context.partition_name] += 1

        for chunk_index, chunk in enumerate(sorted_chunks):
            primitive_steps_total += chunk.primitive_steps
            ensure_same_shape("observation/image", chunk.observation_image, image_shape)
            ensure_same_shape("observation/state", chunk.observation_state, state_shape)
            ensure_same_shape("tokenized_prompt", chunk.tokenized_prompt, token_shape)
            ensure_same_shape("tokenized_prompt_mask", chunk.tokenized_prompt_mask, token_mask_shape)
            ensure_same_shape("nft_xt", chunk.nft_xt, nft_xt_shape)
            ensure_same_shape("nft_v", chunk.nft_v, nft_v_shape)
            ensure_same_shape("nft_xnext", chunk.nft_xnext, nft_xnext_shape)
            ensure_same_shape("nft_step_index", chunk.nft_step_index, nft_step_index_shape)
            ensure_same_shape("nft_noise_level", chunk.nft_noise_level, nft_noise_level_shape)
            if has_wrist_image:
                if chunk.observation_wrist_image is None:
                    wrist_array = np.zeros(wrist_shape, dtype=np.uint8)
                else:
                    ensure_same_shape("observation/wrist_image", chunk.observation_wrist_image, wrist_shape)
                    wrist_array = chunk.observation_wrist_image
                observation_wrist_image[chunk_index, batch_index] = torch.as_tensor(
                    wrist_array,
                    dtype=torch.uint8,
                )
            if prev_logprobs is not None:
                if chunk.prev_logprobs is None:
                    raise ValueError(
                        f"context {chunk.context_id} policy_query_index={chunk.policy_query_index} "
                        "is missing prev_logprobs while --include-prev-logprobs was requested"
                    )
                ensure_same_shape("prev_logprobs", chunk.prev_logprobs, prev_logprob_shape)
                prev_logprobs[chunk_index, batch_index] = torch.as_tensor(
                    chunk.prev_logprobs,
                    dtype=torch.float32,
                )

            rewards[chunk_index, batch_index, 0] = float(chunk.reward_sum)
            success_once[chunk_index, batch_index, 0] = bool(chunk.success)
            dones[chunk_index + 1, batch_index, 0] = bool(chunk.done)
            terminations[chunk_index + 1, batch_index, 0] = bool(chunk.termination)
            truncations[chunk_index + 1, batch_index, 0] = bool(chunk.truncation)
            loss_mask[chunk_index, batch_index, 0] = True

            observation_image[chunk_index, batch_index] = torch.as_tensor(
                chunk.observation_image, dtype=torch.uint8
            )
            observation_state[chunk_index, batch_index] = torch.as_tensor(
                chunk.observation_state, dtype=torch.float32
            )
            tokenized_prompt[chunk_index, batch_index] = torch.as_tensor(
                chunk.tokenized_prompt, dtype=torch.int32
            )
            tokenized_prompt_mask[chunk_index, batch_index] = torch.as_tensor(
                chunk.tokenized_prompt_mask, dtype=torch.bool
            )
            nft_xt[chunk_index, batch_index] = torch.as_tensor(chunk.nft_xt, dtype=torch.float32)
            nft_v[chunk_index, batch_index] = torch.as_tensor(chunk.nft_v, dtype=torch.float32)
            nft_xnext[chunk_index, batch_index] = torch.as_tensor(
                chunk.nft_xnext, dtype=torch.float32
            )
            nft_step_index[chunk_index, batch_index] = torch.as_tensor(
                chunk.nft_step_index, dtype=torch.int64
            )
            nft_noise_level[chunk_index, batch_index] = torch.as_tensor(
                chunk.nft_noise_level, dtype=torch.float32
            )

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True).expand_as(loss_mask).clone()

    payload: dict[str, torch.Tensor] = {
        "rewards": rewards.contiguous(),
        "dones": dones.contiguous(),
        "terminations": terminations.contiguous(),
        "truncations": truncations.contiguous(),
        "success_once": success_once.contiguous(),
        "loss_mask": loss_mask.contiguous(),
        "loss_mask_sum": loss_mask_sum.contiguous(),
        "observation/image": observation_image.contiguous(),
        "observation/state": observation_state.contiguous(),
        "tokenized_prompt": tokenized_prompt.contiguous(),
        "tokenized_prompt_mask": tokenized_prompt_mask.contiguous(),
        "nft_xt": nft_xt.contiguous(),
        "nft_v": nft_v.contiguous(),
        "nft_xnext": nft_xnext.contiguous(),
        "nft_step_index": nft_step_index.contiguous(),
        "nft_noise_level": nft_noise_level.contiguous(),
    }
    if observation_wrist_image is not None:
        payload["observation/wrist_image"] = observation_wrist_image.contiguous()
    if prev_logprobs is not None:
        payload["prev_logprobs"] = prev_logprobs.contiguous()

    summary = {
        "artifact_format": ROLL_OUT_BATCH_FORMAT,
        "contexts_covered": batch_size,
        "trace_records_read": int(sum(context_chunk_counts.values())),
        "primitive_steps_total": primitive_steps_total,
        "max_chunks_per_context": num_chunk,
        "context_chunk_counts": context_chunk_counts,
        "family_counts": dict(family_counts),
        "partition_counts": dict(partition_counts),
        "context_task_names": context_task_names,
        "has_wrist_image": has_wrist_image,
        "includes_prev_logprobs": prev_logprobs is not None,
        "tensor_shapes": {key: list(value.shape) for key, value in payload.items()},
        "notes": [
            "This artifact is chunk-level because the current RLinf NFT LIBERO config uses algorithm.reward_type=chunk_level.",
            "nft_xt and nft_xnext preserve the full OpenPI internal horizon; nft_v carries the chunk-level exact velocity target used by the current Pi0.5 NFT loss path.",
            "success_once and loss_mask are padded per context so terminal-binary advantages can be computed offline without live rollout workers.",
            "prev_values are omitted because the current exact post-train target is NFT actor without a value head.",
        ],
    }
    return payload, summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    trace_path = Path(args.trace_path).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve() if args.summary_path else None
    ensure_parent(output_path)
    if summary_path is not None:
        ensure_parent(summary_path)

    max_token_len, discrete_state_input = resolve_tokenizer_settings(args)
    tokenizer = openpi_tokenizer.PaligemmaTokenizer(max_len=max_token_len)
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    contexts_by_id: collections.OrderedDict[str, ContextPayload] = collections.OrderedDict()
    trace_records_read = 0
    partition_counts_raw: collections.Counter[str] = collections.Counter()
    family_counts_raw: collections.Counter[str] = collections.Counter()

    for record in iter_trace_records(trace_path, args.max_records):
        context_id = str(record["context_id"])
        if context_id not in contexts_by_id:
            if args.max_contexts is not None and len(contexts_by_id) >= args.max_contexts:
                continue
            contexts_by_id[context_id] = ContextPayload(
                context_id=context_id,
                proxy_family_id=(
                    None
                    if record.get("proxy_family_id") is None
                    else str(record["proxy_family_id"])
                ),
                partition_name=(
                    None
                    if record.get("partition_name") is None
                    else str(record["partition_name"])
                ),
                task_id=_normalize_task_id(record.get("task_id")),
                task_name=(
                    None if record.get("task_name") is None else str(record["task_name"])
                ),
                prompt=str(record["prompt"]),
            )
        context = contexts_by_id[context_id]
        context.chunks.append(
            extract_chunk_payload(
                record,
                tokenizer=tokenizer,
                discrete_state_input=discrete_state_input,
                token_cache=token_cache,
            )
        )
        trace_records_read += 1
        if context.partition_name is not None:
            partition_counts_raw[context.partition_name] += 1
        if context.proxy_family_id is not None:
            family_counts_raw[context.proxy_family_id] += 1

    contexts = list(contexts_by_id.values())
    payload, payload_summary = build_rollout_payload(
        contexts,
        include_prev_logprobs=bool(args.include_prev_logprobs),
    )

    torch.save(payload, output_path)
    logging.info("wrote offline rollout batch to %s", output_path)

    summary = {
        "trace_path": str(trace_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "openpi_config_name": args.openpi_config_name,
        "max_token_len": max_token_len,
        "discrete_state_input": discrete_state_input,
        "max_records": args.max_records,
        "max_contexts": args.max_contexts,
        "trace_records_read": trace_records_read,
        "raw_partition_record_counts": dict(partition_counts_raw),
        "raw_family_record_counts": dict(family_counts_raw),
    }
    summary.update(payload_summary)

    if summary_path is not None:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        logging.info("wrote conversion summary to %s", summary_path)


if __name__ == "__main__":
    main()
