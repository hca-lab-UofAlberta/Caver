#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from openpi_client import websocket_client_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end websocket policy latency with synthetic observations."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--prompt", default="latency probe")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": float(sum(values) / len(values)),
        "min_ms": float(min(values)),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": float(max(values)),
    }


def build_observation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "observation/image": np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros(
            (args.image_size, args.image_size, 3), dtype=np.uint8
        ),
        "observation/state": np.zeros((args.state_dim,), dtype=np.float32),
        "prompt": args.prompt,
    }


def extract_server_infer_ms(result: dict[str, Any]) -> float | None:
    timing = result.get("server_timing")
    if not isinstance(timing, dict):
        return None
    infer_ms = timing.get("infer_ms")
    if infer_ms is None:
        return None
    try:
        return float(infer_ms)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = policy.get_server_metadata()
    observation = build_observation(args)

    for _ in range(max(args.warmup, 0)):
        policy.infer(observation)

    client_latencies_ms: list[float] = []
    server_latencies_ms: list[float] = []
    action_shape: tuple[int, ...] | None = None

    for _ in range(max(args.requests, 0)):
        start = time.perf_counter()
        result = policy.infer(observation)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        client_latencies_ms.append(elapsed_ms)

        if action_shape is None:
            action_shape = tuple(np.asarray(result["actions"]).shape)

        server_infer_ms = extract_server_infer_ms(result)
        if server_infer_ms is not None:
            server_latencies_ms.append(server_infer_ms)

    payload = {
        "metadata": metadata,
        "requests": args.requests,
        "warmup": args.warmup,
        "action_shape": action_shape,
        "client_latency_ms": summarize(client_latencies_ms),
        "server_infer_ms": summarize(server_latencies_ms),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"OK metadata {metadata}")
    print(f"OK action_shape={action_shape}")
    print(f"OK client_latency_ms={payload['client_latency_ms']}")
    if server_latencies_ms:
        print(f"OK server_infer_ms={payload['server_infer_ms']}")
    else:
        print("OK server_infer_ms=unavailable")


if __name__ == "__main__":
    main()
