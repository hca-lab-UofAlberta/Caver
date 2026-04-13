#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
from datetime import datetime
from typing import Any


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[ray-probe {timestamp}] {message}", flush=True)


def wrap_call(module: Any, attr: str, label: str) -> None:
    original = getattr(module, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        log(f"{label}: start")
        result = original(*args, **kwargs)
        log(f"{label}: done")
        return result

    setattr(module, attr, wrapped)


def wrap_thread_start() -> None:
    original = threading.Thread.start

    def wrapped(self: threading.Thread, *args: Any, **kwargs: Any) -> Any:
        log(f"thread.start({self.name}): start")
        result = original(self, *args, **kwargs)
        log(f"thread.start({self.name}): done")
        return result

    threading.Thread.start = wrapped


def wrap_raylet_constructor(raylet_module: Any, attr: str) -> None:
    original = getattr(raylet_module, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        log(f"{attr}: start")
        result = original(*args, **kwargs)
        log(f"{attr}: done")
        return result

    setattr(raylet_module, attr, wrapped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instrumented local ray.init() probe")
    parser.add_argument("--namespace", default="RLinf")
    parser.add_argument(
        "--include-dashboard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--log-to-driver",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--node-ip-address", default=None)
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")

    import ray
    import ray._private.node
    import ray._private.state
    import ray._private.worker
    import ray._raylet

    wrap_call(ray._private.worker, "connect", "worker.connect")
    wrap_call(ray._private.worker, "_initialize_internal_kv", "_initialize_internal_kv")
    wrap_call(
        ray._private.state.state,
        "_initialize_global_state",
        "state._initialize_global_state",
    )
    wrap_call(ray._private.node.Node, "get_gcs_client", "Node.get_gcs_client")
    wrap_call(
        ray._private.node.Node,
        "check_version_info",
        "Node.check_version_info",
    )
    wrap_call(
        ray._private.node.Node,
        "should_redirect_logs",
        "Node.should_redirect_logs",
    )
    wrap_raylet_constructor(ray._raylet, "CoreWorker")
    wrap_raylet_constructor(ray._raylet, "GcsErrorSubscriber")
    wrap_raylet_constructor(ray._raylet, "GcsLogSubscriber")
    wrap_thread_start()

    init_kwargs: dict[str, Any] = {
        "include_dashboard": args.include_dashboard,
        "logging_level": "INFO",
        "namespace": args.namespace,
        "log_to_driver": args.log_to_driver,
        "local_mode": args.local_mode,
    }
    if args.node_ip_address:
        init_kwargs["_node_ip_address"] = args.node_ip_address
    if args.num_cpus is not None:
        init_kwargs["num_cpus"] = args.num_cpus
    if args.num_gpus is not None:
        init_kwargs["num_gpus"] = args.num_gpus

    log(f"before ray.init kwargs={init_kwargs}")
    ctx = ray.init(**init_kwargs)
    log("after ray.init")
    log(
        "address_info="
        + repr(
            {
                key: ctx.address_info.get(key)
                for key in (
                    "node_ip_address",
                    "gcs_address",
                    "raylet_ip_address",
                    "webui_url",
                    "session_dir",
                )
            }
        )
    )
    ray.shutdown()
    log("after ray.shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
