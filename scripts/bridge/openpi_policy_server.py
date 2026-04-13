#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np


class DummyPolicy:
    def __init__(self, action_horizon: int, action_dim: int) -> None:
        self._action_horizon = action_horizon
        self._action_dim = action_dim

    def infer(self, _obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "actions": np.zeros((self._action_horizon, self._action_dim), dtype=np.float32),
            "state": np.zeros((self._action_dim,), dtype=np.float32),
        }


def _strip_single_batch(value: Any) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is expected in real policy mode
        torch = None

    if isinstance(value, dict):
        return {key: _strip_single_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_single_batch(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_single_batch(item) for item in value]
    if torch is not None and torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        value = np.asarray(tensor)
    elif isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == 1:
            value = value[0]
        return value

    return value


def _default_rlinf_config_name(openpi_config_name: str) -> str:
    config_map = {
        "pi0_libero": "libero_goal_ppo_openpi",
        "pi05_libero": "libero_goal_ppo_openpi_pi05",
    }
    if openpi_config_name not in config_map:
        supported = ", ".join(sorted(config_map))
        raise SystemExit(
            "error: --exact-rollout-payload requires --rlinf-config-name for "
            f"unsupported OpenPI config {openpi_config_name!r}. Supported implicit mappings: {supported}"
        )
    return config_map[openpi_config_name]


class RLinfExactPolicy:
    def __init__(self, model, *, infer_mode: str) -> None:
        self._model = model
        self._model.eval()
        self._infer_mode = infer_mode

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        import torch

        image = np.array(obs["observation/image"], copy=True)
        wrist_image = obs.get("observation/wrist_image")
        state = np.array(obs["observation/state"], dtype=np.float32, copy=True)
        prompt = str(obs.get("prompt", ""))
        if wrist_image is not None:
            wrist_image = np.array(wrist_image, copy=True)

        env_obs = {
            "main_images": torch.from_numpy(np.ascontiguousarray(image[None, ...])),
            "states": torch.from_numpy(np.ascontiguousarray(state[None, ...])),
            "task_descriptions": [prompt],
            "wrist_images": (
                torch.from_numpy(np.ascontiguousarray(wrist_image[None, ...]))
                if wrist_image is not None
                else None
            ),
        }

        with torch.no_grad():
            actions, result = self._model.predict_action_batch(
                env_obs=env_obs,
                mode=self._infer_mode,
                compute_values=True,
                return_obs=True,
            )

        payload = {
            "actions": np.asarray(actions[0], dtype=np.float32),
        }
        payload.update(
            {key: _strip_single_batch(value) for key, value in result.items()}
        )
        return payload


def build_rlinf_exact_policy(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from rlinf.models import get_model

    embodied_path = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "src"
        / "pi-StepNFT"
        / "examples"
        / "embodiment"
    )
    os.environ.setdefault("EMBODIED_PATH", str(embodied_path))
    rlinf_config_name = args.rlinf_config_name or _default_rlinf_config_name(
        args.config_name
    )
    config_dir = embodied_path / "config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        full_cfg = compose(config_name=rlinf_config_name)

    actor_model_cfg = full_cfg.actor.model
    with open_dict(actor_model_cfg):
        actor_model_cfg.model_path = args.pretrained_path
        actor_model_cfg.model_type = "openpi"
        actor_model_cfg.num_steps = args.num_steps
        actor_model_cfg.action_dim = args.action_dim
        actor_model_cfg.is_lora = bool(actor_model_cfg.get("is_lora", False))
        if args.exact_action_chunk is not None:
            actor_model_cfg.num_action_chunks = args.exact_action_chunk
        actor_model_cfg.openpi.config_name = args.config_name
        actor_model_cfg.openpi.num_steps = args.num_steps
        actor_model_cfg.openpi.action_env_dim = args.action_dim
        actor_model_cfg.openpi.use_nft_loss = not args.exact_no_nft_loss
        actor_model_cfg.openpi.solver_type = args.exact_solver_type
        if args.exact_action_chunk is not None:
            actor_model_cfg.openpi.action_chunk = args.exact_action_chunk
        if args.exact_add_value_head:
            actor_model_cfg.add_value_head = True
            actor_model_cfg.openpi.add_value_head = True
        if args.exact_value_after_vlm:
            actor_model_cfg.openpi.value_after_vlm = True

    model = get_model(actor_model_cfg)
    metadata = {
        "mode": "rlinf_exact_policy",
        "config_name": args.config_name,
        "pretrained_path": args.pretrained_path,
        "rlinf_config_name": rlinf_config_name,
        "num_steps": args.num_steps,
        "action_dim": args.action_dim,
        "exact_action_chunk": int(actor_model_cfg.openpi.action_chunk),
        "exact_use_nft_loss": bool(actor_model_cfg.openpi.use_nft_loss),
        "exact_add_value_head": bool(actor_model_cfg.openpi.add_value_head),
        "exact_value_after_vlm": bool(actor_model_cfg.openpi.value_after_vlm),
        "exact_solver_type": str(actor_model_cfg.openpi.solver_type),
        "exact_infer_mode": args.exact_infer_mode,
    }
    return RLinfExactPolicy(model, infer_mode=args.exact_infer_mode), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an OpenPI policy over websockets.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dummy", action="store_true", help="Serve a zero-action dummy policy.")
    parser.add_argument("--action-horizon", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--exact-rollout-payload",
        action="store_true",
        help=(
            "Serve the policy through RLinf's OpenPI model path and return exact rollout-time "
            "payloads such as prev_logprobs/prev_values/forward_inputs."
        ),
    )
    parser.add_argument(
        "--rlinf-config-name",
        default=None,
        help=(
            "Optional RLinf Hydra config name used to compose actor.model for "
            "--exact-rollout-payload. Defaults to a built-in LIBERO mapping from --config-name."
        ),
    )
    parser.add_argument(
        "--exact-action-chunk",
        type=int,
        default=None,
        help="Optional override for actor.model.num_action_chunks / openpi.action_chunk in exact mode.",
    )
    parser.add_argument(
        "--exact-no-nft-loss",
        action="store_true",
        help="Disable NFT rollout traces in exact mode. By default, exact mode enables use_nft_loss.",
    )
    parser.add_argument(
        "--exact-add-value-head",
        action="store_true",
        help="Enable add_value_head in exact mode even if the composed RLinf config disables it.",
    )
    parser.add_argument(
        "--exact-value-after-vlm",
        action="store_true",
        help="Enable value_after_vlm in exact mode when combined with a value head.",
    )
    parser.add_argument(
        "--exact-solver-type",
        default="flow_sde",
        help=(
            "OpenPI solver_type override in exact mode. "
            "Use flow_sde to emit nft_* flow snapshots needed by the exact rollout converter."
        ),
    )
    parser.add_argument(
        "--exact-infer-mode",
        default="train",
        choices=("train", "eval"),
        help=(
            "predict_action_batch mode for exact inference. "
            "Use train to preserve NFT flow snapshots; eval forces Euler sampling for NFT models."
        ),
    )
    return parser.parse_args()


def build_policy(args: argparse.Namespace):
    if args.dummy:
        return DummyPolicy(args.action_horizon, args.action_dim), {
            "mode": "dummy",
            "action_horizon": args.action_horizon,
            "action_dim": args.action_dim,
        }

    if not args.config_name or not args.pretrained_path:
        raise SystemExit(
            "error: real policy mode requires --config-name and --pretrained-path"
        )

    if args.exact_rollout_payload:
        return build_rlinf_exact_policy(args)

    from toolkits.eval_scripts_openpi import setup_policy

    policy_args = argparse.Namespace(
        config_name=args.config_name,
        pretrained_path=args.pretrained_path,
        num_steps=args.num_steps,
    )
    policy = setup_policy(policy_args)
    metadata = {
        "mode": "policy",
        "config_name": args.config_name,
        "pretrained_path": args.pretrained_path,
        "num_steps": args.num_steps,
    }
    return policy, metadata


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy, metadata = build_policy(args)
    metadata["server"] = dataclasses.asdict(
        dataclasses.make_dataclass(
            "ServerConfig", [("host", str), ("port", int)]
        )(args.host, args.port)
    )
    logging.info("starting websocket policy server on %s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
