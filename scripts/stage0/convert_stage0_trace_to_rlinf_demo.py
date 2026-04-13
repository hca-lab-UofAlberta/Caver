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


@dataclass
class ConversionState:
    args: argparse.Namespace
    trace_path: Path
    output_path: Path
    summary_path: Path | None
    max_token_len: int
    discrete_state_input: bool
    trace_records_read: int = 0
    demo_items_written: int = 0
    primitive_steps_total: int = 0
    family_counts: collections.Counter = field(default_factory=collections.Counter)
    partition_counts: collections.Counter = field(default_factory=collections.Counter)
    completed_reason_counts: collections.Counter = field(default_factory=collections.Counter)
    chunk_action_horizons: set[int] = field(default_factory=set)
    context_ids: set[str] = field(default_factory=set)
    pending_items: list[dict[str, Any]] = field(default_factory=list)
    pending_item_context_ids: list[str] = field(default_factory=list)
    shards: list[dict[str, Any]] = field(default_factory=list)

    def record_trace_metadata(self, record: dict[str, Any]) -> None:
        self.trace_records_read += 1
        self.context_ids.add(str(record["context_id"]))
        self.primitive_steps_total += len(record["actions"])
        if record.get("proxy_family_id") is not None:
            self.family_counts[str(record["proxy_family_id"])] += 1
        if record.get("partition_name") is not None:
            self.partition_counts[str(record["partition_name"])] += 1
        completed_reason = record.get("completed_reason")
        self.completed_reason_counts[str(completed_reason or "unknown")] += 1
        self.chunk_action_horizons.add(int(record["chunk_action_horizon"]))

    def add_demo_items(self, demo_items: list[dict[str, Any]], *, context_id: str) -> None:
        self.pending_items.extend(demo_items)
        self.pending_item_context_ids.extend([context_id] * len(demo_items))
        self.demo_items_written += len(demo_items)
        self.flush_shards_if_needed()

    def flush_shards_if_needed(self) -> None:
        if self.args.output_mode != "sharded_manifest":
            return
        shard_limit = int(self.args.max_items_per_shard)
        while len(self.pending_items) >= shard_limit:
            self._write_shard(
                self.pending_items[:shard_limit],
                self.pending_item_context_ids[:shard_limit],
            )
            del self.pending_items[:shard_limit]
            del self.pending_item_context_ids[:shard_limit]

    def finalize_output(self) -> None:
        if not self.demo_items_written:
            raise ValueError("conversion produced zero demo items")
        if self.args.output_mode == "single_pt":
            torch.save(self.pending_items, self.output_path)
            logging.info("wrote demo artifact to %s", self.output_path)
            return

        if self.pending_items:
            self._write_shard(self.pending_items, self.pending_item_context_ids)
            self.pending_items = []
            self.pending_item_context_ids = []
        manifest = self.build_manifest()
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logging.info("wrote sharded demo manifest to %s", self.output_path)

    def build_manifest(self) -> dict[str, Any]:
        return {
            "artifact_format": "robot_demo_sharded",
            "format_version": 1,
            "trace_path": str(self.trace_path),
            "manifest_path": str(self.output_path),
            "demo_format": self.args.demo_format,
            "openpi_config_name": self.args.openpi_config_name,
            "max_token_len": self.max_token_len,
            "discrete_state_input": self.discrete_state_input,
            "total_items": self.demo_items_written,
            "trace_records_read": self.trace_records_read,
            "contexts_covered": len(self.context_ids),
            "max_items_per_shard": int(self.args.max_items_per_shard),
            "shards": self.shards,
        }

    def build_summary(self) -> dict[str, Any]:
        summary = {
            "trace_path": str(self.trace_path),
            "output_path": str(self.output_path),
            "summary_path": str(self.summary_path) if self.summary_path is not None else None,
            "output_mode": self.args.output_mode,
            "demo_format": self.args.demo_format,
            "openpi_config_name": self.args.openpi_config_name,
            "max_token_len": self.max_token_len,
            "discrete_state_input": self.discrete_state_input,
            "trace_records_read": self.trace_records_read,
            "demo_items_written": self.demo_items_written,
            "contexts_covered": len(self.context_ids),
            "primitive_steps_total": self.primitive_steps_total,
            "family_counts": dict(self.family_counts),
            "partition_counts": dict(self.partition_counts),
            "completed_reason_counts": dict(self.completed_reason_counts),
            "chunk_action_horizons": sorted(self.chunk_action_horizons),
            "notes": [
                "This artifact targets the public pi-StepNFT robot_demo -> SACReplayBuffer path.",
                "The stock public NFT actor path still lacks a native offline demo ingest hook.",
            ],
        }
        if self.args.output_mode == "sharded_manifest":
            summary["total_shards"] = len(self.shards)
            summary["max_items_per_shard"] = int(self.args.max_items_per_shard)
        return summary

    def _write_shard(self, demo_items: list[dict[str, Any]], context_ids: list[str]) -> None:
        shard_dir = self.output_path.parent / f"{self.output_path.stem}_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_index = len(self.shards)
        shard_path = shard_dir / f"demo_shard_{shard_index:05d}.pt"
        torch.save(demo_items, shard_path)
        self.shards.append(
            {
                "path": str(shard_path.relative_to(self.output_path.parent)),
                "items": len(demo_items),
                "contexts_covered": len(set(context_ids)),
            }
        )
        logging.info(
            "wrote shard %s (%d items, %d contexts)",
            shard_path,
            len(demo_items),
            len(set(context_ids)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Stage-0 chunk traces into a pi-StepNFT SAC replay-buffer demo artifact."
    )
    parser.add_argument("--trace-path", required=True, help="Chunk-trace JSONL path from libero_remote_eval.py")
    parser.add_argument("--output-path", required=True, help="Output .pt path or sharded manifest .json path")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON summary path describing the converted artifact",
    )
    parser.add_argument(
        "--output-mode",
        default="single_pt",
        choices=("single_pt", "sharded_manifest"),
        help="single_pt writes one .pt list; sharded_manifest writes .pt shards plus a JSON manifest",
    )
    parser.add_argument(
        "--max-items-per-shard",
        type=int,
        default=256,
        help="Maximum demo items per shard when --output-mode=sharded_manifest",
    )
    parser.add_argument(
        "--demo-format",
        default="chunk_step",
        choices=("chunk_step", "primitive_step"),
        help="chunk_step keeps one replay item per policy query; primitive_step expands each executed action",
    )
    parser.add_argument(
        "--openpi-config-name",
        default="pi05_libero",
        help="OpenPI config name used to recover prompt-tokenizer settings",
    )
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=None,
        help="Optional override for prompt token length. Defaults to the OpenPI config's max_token_len.",
    )
    parser.add_argument(
        "--discrete-state-input",
        action="store_true",
        help="Tokenize prompts with Pi0.5-style discretized state text instead of prompt-only mode.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on the number of chunk records read from the trace",
    )
    args = parser.parse_args()
    if args.output_mode == "sharded_manifest" and args.max_items_per_shard <= 0:
        parser.error("--max-items-per-shard must be positive")
    return args


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
            "failed to resolve tokenizer settings from config %s: %s; falling back to heuristic defaults",
            args.openpi_config_name,
            exc,
        )
        return (200 if "pi05" in args.openpi_config_name else 48), discrete_state_input


def _iter_jsonl_trace_records(
    trace_path: Path, max_records: int | None
) -> Iterator[dict[str, Any]]:
    produced = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
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
                break
    if produced == 0:
        raise ValueError(f"trace path produced zero records: {trace_path}")


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
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
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


def iter_trace_records(
    trace_path: Path, max_records: int | None
) -> Iterator[dict[str, Any]]:
    manifest = load_trace_source_manifest(trace_path)
    if manifest is None:
        yield from _iter_jsonl_trace_records(trace_path, max_records)
        return

    produced = 0
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
    if not discrete_state_input:
        token_cache[prompt] = (tokens, token_mask)
    return tokens, token_mask


def build_tensor_observation(
    obs_payload: dict[str, Any],
    prompt: str,
    *,
    tokenizer: openpi_tokenizer.PaligemmaTokenizer,
    discrete_state_input: bool,
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, torch.Tensor]:
    state = np.asarray(obs_payload["state"], dtype=np.float32)
    tokens, token_mask = tokenize_prompt(
        tokenizer,
        prompt,
        state,
        discrete_state_input=discrete_state_input,
        token_cache=token_cache,
    )
    observation = {
        "observation/image": torch.as_tensor(
            np.asarray(obs_payload["image"], dtype=np.uint8), dtype=torch.uint8
        ),
        "observation/state": torch.as_tensor(state, dtype=torch.float32),
        "tokenized_prompt": torch.as_tensor(tokens, dtype=torch.int32),
        "tokenized_prompt_mask": torch.as_tensor(token_mask, dtype=torch.bool),
    }
    wrist_image = obs_payload.get("wrist_image")
    if wrist_image is not None:
        observation["observation/wrist_image"] = torch.as_tensor(
            np.asarray(wrist_image, dtype=np.uint8), dtype=torch.uint8
        )
    return observation


def pad_vector(
    values: np.ndarray,
    *,
    length: int,
    pad_value: float | bool,
    dtype: np.dtype,
) -> np.ndarray:
    if values.shape[0] > length:
        raise ValueError(f"cannot pad vector of length {values.shape[0]} into shorter length {length}")
    if values.shape[0] == length:
        return values.astype(dtype, copy=False)
    padded = np.full((length,), pad_value, dtype=dtype)
    padded[: values.shape[0]] = values.astype(dtype, copy=False)
    return padded


def pad_actions(actions: np.ndarray, *, horizon: int) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"expected action array with shape [steps, action_dim], got {actions.shape}")
    if actions.shape[0] > horizon:
        raise ValueError(f"cannot pad {actions.shape[0]} executed steps into horizon {horizon}")
    if actions.shape[0] == horizon:
        return actions.astype(np.float32, copy=False)
    padded = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    padded[: actions.shape[0]] = actions.astype(np.float32, copy=False)
    return padded


def validate_chunk_record(record: dict[str, Any]) -> None:
    steps_executed = len(record["actions"])
    expected_lengths = {
        "rewards": len(record["rewards"]),
        "dones": len(record["dones"]),
        "terminations": len(record["terminations"]),
        "truncations": len(record["truncations"]),
        "success_once": len(record["success_once"]),
        "next_obs_sequence": len(record["next_obs_sequence"]),
    }
    bad_lengths = {key: value for key, value in expected_lengths.items() if value != steps_executed}
    if bad_lengths:
        raise ValueError(
            f"trace record {record['context_id']} policy_query_index={record['policy_query_index']} "
            f"has inconsistent lengths: actions={steps_executed}, {bad_lengths}"
        )


def build_chunk_demo_item(
    record: dict[str, Any],
    *,
    tokenizer: openpi_tokenizer.PaligemmaTokenizer,
    discrete_state_input: bool,
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    validate_chunk_record(record)
    prompt = str(record["prompt"])
    horizon = int(record["chunk_action_horizon"])
    actions = pad_actions(np.asarray(record["actions"], dtype=np.float32), horizon=horizon)
    rewards = pad_vector(
        np.asarray(record["rewards"], dtype=np.float32),
        length=horizon,
        pad_value=0.0,
        dtype=np.float32,
    )
    dones = pad_vector(
        np.asarray(record["dones"], dtype=np.bool_),
        length=horizon,
        pad_value=bool(record["dones"][-1]) if record["dones"] else False,
        dtype=np.bool_,
    )
    terminations = pad_vector(
        np.asarray(record["terminations"], dtype=np.bool_),
        length=horizon,
        pad_value=bool(record["terminations"][-1]) if record["terminations"] else False,
        dtype=np.bool_,
    )
    truncations = pad_vector(
        np.asarray(record["truncations"], dtype=np.bool_),
        length=horizon,
        pad_value=bool(record["truncations"][-1]) if record["truncations"] else False,
        dtype=np.bool_,
    )
    success_once = pad_vector(
        np.asarray(record["success_once"], dtype=np.bool_),
        length=horizon,
        pad_value=bool(record["success_once"][-1]) if record["success_once"] else False,
        dtype=np.bool_,
    )

    return {
        "action": torch.as_tensor(actions.reshape(-1), dtype=torch.float32),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32),
        "dones": torch.as_tensor(dones, dtype=torch.bool),
        "terminations": torch.as_tensor(terminations, dtype=torch.bool),
        "truncations": torch.as_tensor(truncations, dtype=torch.bool),
        "success_once": torch.as_tensor(success_once, dtype=torch.bool),
        "transitions": {
            "obs": build_tensor_observation(
                record["obs"],
                prompt,
                tokenizer=tokenizer,
                discrete_state_input=discrete_state_input,
                token_cache=token_cache,
            ),
            "next_obs": build_tensor_observation(
                record["next_obs_sequence"][-1],
                prompt,
                tokenizer=tokenizer,
                discrete_state_input=discrete_state_input,
                token_cache=token_cache,
            ),
        },
    }


def build_primitive_demo_items(
    record: dict[str, Any],
    *,
    tokenizer: openpi_tokenizer.PaligemmaTokenizer,
    discrete_state_input: bool,
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    validate_chunk_record(record)
    prompt = str(record["prompt"])
    actions = np.asarray(record["actions"], dtype=np.float32)
    rewards = np.asarray(record["rewards"], dtype=np.float32)
    dones = np.asarray(record["dones"], dtype=np.bool_)
    terminations = np.asarray(record["terminations"], dtype=np.bool_)
    truncations = np.asarray(record["truncations"], dtype=np.bool_)
    success_once = np.asarray(record["success_once"], dtype=np.bool_)
    next_obs_sequence = list(record["next_obs_sequence"])

    items: list[dict[str, Any]] = []
    current_obs = record["obs"]
    for step_index in range(actions.shape[0]):
        next_obs = next_obs_sequence[step_index]
        items.append(
            {
                "action": torch.as_tensor(actions[step_index].reshape(-1), dtype=torch.float32),
                "rewards": torch.as_tensor([rewards[step_index]], dtype=torch.float32),
                "dones": torch.as_tensor([bool(dones[step_index])], dtype=torch.bool),
                "terminations": torch.as_tensor([bool(terminations[step_index])], dtype=torch.bool),
                "truncations": torch.as_tensor([bool(truncations[step_index])], dtype=torch.bool),
                "success_once": torch.as_tensor([bool(success_once[step_index])], dtype=torch.bool),
                "transitions": {
                    "obs": build_tensor_observation(
                        current_obs,
                        prompt,
                        tokenizer=tokenizer,
                        discrete_state_input=discrete_state_input,
                        token_cache=token_cache,
                    ),
                    "next_obs": build_tensor_observation(
                        next_obs,
                        prompt,
                        tokenizer=tokenizer,
                        discrete_state_input=discrete_state_input,
                        token_cache=token_cache,
                    ),
                },
            }
        )
        current_obs = next_obs
    return items


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    trace_path = Path(args.trace_path).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve() if args.summary_path else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    max_token_len, discrete_state_input = resolve_tokenizer_settings(args)
    tokenizer = openpi_tokenizer.PaligemmaTokenizer(max_len=max_token_len)
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    state = ConversionState(
        args=args,
        trace_path=trace_path,
        output_path=output_path,
        summary_path=summary_path,
        max_token_len=max_token_len,
        discrete_state_input=discrete_state_input,
    )

    for record in iter_trace_records(trace_path, args.max_records):
        state.record_trace_metadata(record)
        context_id = str(record["context_id"])
        if args.demo_format == "chunk_step":
            demo_items = [
                build_chunk_demo_item(
                    record,
                    tokenizer=tokenizer,
                    discrete_state_input=discrete_state_input,
                    token_cache=token_cache,
                )
            ]
        else:
            demo_items = build_primitive_demo_items(
                record,
                tokenizer=tokenizer,
                discrete_state_input=discrete_state_input,
                token_cache=token_cache,
            )
        state.add_demo_items(demo_items, context_id=context_id)

    state.finalize_output()
    summary = state.build_summary()
    if summary_path is not None:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logging.info("wrote conversion summary to %s", summary_path)


if __name__ == "__main__":
    main()
