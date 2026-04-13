#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from openpi_client import websocket_client_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the OpenPI websocket policy bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = policy.get_server_metadata()
    print(f"OK metadata {metadata}")

    observation = {
        "observation/image": np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8),
        "observation/state": np.zeros((args.state_dim,), dtype=np.float32),
        "prompt": "smoke test",
    }
    result = policy.infer(observation)
    actions = np.asarray(result["actions"])
    print(f"OK actions shape={actions.shape} dtype={actions.dtype}")
    extra_keys = sorted(key for key in result.keys() if key != "actions")
    if extra_keys:
        print(f"OK extra_keys={extra_keys}")
        for key in extra_keys:
            value = result[key]
            if isinstance(value, dict):
                print(f"OK {key}=dict keys={sorted(value.keys())}")
            else:
                array = np.asarray(value)
                print(f"OK {key} shape={array.shape} dtype={array.dtype}")


if __name__ == "__main__":
    main()
