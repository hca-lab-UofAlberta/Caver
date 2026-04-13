#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


PARTITION_ORDER = ("T_seed_S0", "T_val_S0", "T_test_S0")
DEFAULT_PARTITION_COUNTS = {
    "T_seed_S0": 24,
    "T_val_S0": 20,
    "T_test_S0": 20,
}


@dataclass(frozen=True)
class ContextRecord:
    context_id: str
    suite: str
    task_index: int
    task_name: str
    init_state_index: int
    proxy_family_id: str
    proposal_task: str

    def to_json(self) -> Dict[str, object]:
        return {
            "context_id": self.context_id,
            "suite": self.suite,
            "task_index": self.task_index,
            "task_name": self.task_name,
            "init_state_index": self.init_state_index,
            "proxy_family_id": self.proxy_family_id,
            "proposal_task": self.proposal_task,
        }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_spec = repo_root / "metadata/stage0/libero_stage0_task_families.json"
    default_output = repo_root / "metadata/stage0/libero_stage0_partitions.json"

    parser = argparse.ArgumentParser(
        description="Generate deterministic Stage-0 LIBERO partition manifests."
    )
    parser.add_argument("--spec", type=Path, default=default_spec)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_libero(repo_root: Path):
    os.environ.setdefault(
        "LIBERO_CONFIG_PATH", str(repo_root / "third_party/config/libero")
    )
    libero_src = repo_root / "third_party/src/LIBERO"
    if str(libero_src) not in sys.path:
        sys.path.insert(0, str(libero_src))

    from libero.libero import benchmark  # pylint: disable=import-outside-toplevel

    return benchmark


def stable_shuffle(values: List[int], seed_text: str) -> List[int]:
    rng = random.Random(seed_text)
    shuffled = list(values)
    rng.shuffle(shuffled)
    return shuffled


def build_contexts(
    benchmark_module,
    family_id: str,
    proposal_task: str,
    member: Dict[str, object],
    seed: int,
    bench_cache: Dict[str, object],
) -> List[ContextRecord]:
    suite = str(member["suite"])
    task_index = int(member["task_index"])
    expected_name = str(member["task_name"])

    if suite not in bench_cache:
        bench_cache[suite] = benchmark_module.get_benchmark(suite)()
    bench = bench_cache[suite]

    task = bench.get_task(task_index)
    if task.name != expected_name:
        raise ValueError(
            f"task mismatch for {suite}[{task_index}]: "
            f"expected {expected_name!r}, got {task.name!r}"
        )

    init_states = bench.get_task_init_states(task_index)
    available = len(init_states)
    shuffled_indices = stable_shuffle(
        list(range(available)), f"{seed}:{family_id}:{suite}:{task_index}"
    )

    return [
        ContextRecord(
            context_id=(
                f"{family_id}__{suite}__task{task_index:02d}__init{init_idx:03d}"
            ),
            suite=suite,
            task_index=task_index,
            task_name=task.name,
            init_state_index=init_idx,
            proxy_family_id=family_id,
            proposal_task=proposal_task,
        )
        for init_idx in shuffled_indices
    ]


def take_round_robin(
    pools: Dict[str, List[ContextRecord]], count: int, start_offset: int
) -> List[ContextRecord]:
    member_keys = list(pools)
    ordered_keys = member_keys[start_offset:] + member_keys[:start_offset]
    selected: List[ContextRecord] = []

    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if not pools[key]:
                continue
            selected.append(pools[key].pop(0))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(
                f"unable to allocate {count} contexts from the remaining task pools"
            )

    return selected


def count_by_member(contexts: Iterable[ContextRecord]) -> Dict[str, int]:
    counter = Counter(
        f"{record.suite}[{record.task_index}]::{record.task_name}" for record in contexts
    )
    return dict(sorted(counter.items()))


def ensure_disjoint(partitions: Dict[str, List[ContextRecord]]) -> None:
    seen: set[str] = set()
    for partition_name, records in partitions.items():
        for record in records:
            if record.context_id in seen:
                raise ValueError(
                    f"duplicate context {record.context_id} detected in {partition_name}"
                )
            seen.add(record.context_id)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_module = bootstrap_libero(repo_root)
    spec = load_json(args.spec)
    families = spec["families"]

    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/stage0/generate_stage0_partitions.py",
        "generator_seed": args.seed,
        "source_spec": str(args.spec),
        "allocation_policy": {
            "partition_counts_per_family": DEFAULT_PARTITION_COUNTS,
            "train_partition": "T_train_S0 receives all remaining disjoint contexts",
            "allocator": "stable per-task shuffle plus round-robin consumption",
        },
        "notes": [
            "Single stock LIBERO tasks on this installation expose 50 init-state templates each.",
            "The proposal's 24-seed plus 20-val plus 20-test split is therefore enforced at the semantic proxy-family level rather than per single concrete task.",
        ],
        "families": [],
        "global_partition_counts": {},
    }

    bench_cache: Dict[str, object] = {}
    global_counts = Counter()

    for family_index, family in enumerate(families):
        family_id = str(family["family_id"])
        proposal_task = str(family["proposal_task"])
        members = family["members"]

        member_context_pools: Dict[str, List[ContextRecord]] = {}
        member_descriptors = []
        total_available = 0

        for member in members:
            contexts = build_contexts(
                benchmark_module=benchmark_module,
                family_id=family_id,
                proposal_task=proposal_task,
                member=member,
                seed=args.seed,
                bench_cache=bench_cache,
            )
            key = f"{member['suite']}[{member['task_index']}]"
            member_context_pools[key] = contexts
            total_available += len(contexts)
            member_descriptors.append(
                {
                    "suite": member["suite"],
                    "task_index": member["task_index"],
                    "task_name": member["task_name"],
                    "available_contexts": len(contexts),
                }
            )

        partitions: Dict[str, List[ContextRecord]] = {}
        for partition_offset, partition_name in enumerate(PARTITION_ORDER):
            partitions[partition_name] = take_round_robin(
                member_context_pools,
                DEFAULT_PARTITION_COUNTS[partition_name],
                start_offset=(family_index + partition_offset) % len(member_context_pools),
            )

        partitions["T_train_S0"] = []
        for member_key in member_context_pools:
            partitions["T_train_S0"].extend(member_context_pools[member_key])

        ensure_disjoint(partitions)

        family_partition_counts = {
            name: len(records) for name, records in partitions.items()
        }
        if family_partition_counts["T_train_S0"] <= 0:
            raise ValueError(f"{family_id} has no remaining train contexts")

        for partition_name, count in family_partition_counts.items():
            global_counts[partition_name] += count

        output["families"].append(
            {
                "family_id": family_id,
                "proposal_task": proposal_task,
                "proxy_label": family["proxy_label"],
                "substitution_note": family["substitution_note"],
                "member_tasks": member_descriptors,
                "available_contexts": total_available,
                "partition_counts": family_partition_counts,
                "partition_member_breakdown": {
                    name: count_by_member(records)
                    for name, records in partitions.items()
                },
                "partitions": {
                    name: [record.to_json() for record in records]
                    for name, records in partitions.items()
                },
            }
        )

    output["global_partition_counts"] = dict(sorted(global_counts.items()))
    output["max_stage0_budget_supported"] = output["global_partition_counts"]["T_train_S0"]
    output["proposal_max_budget_supported"] = (
        output["global_partition_counts"]["T_train_S0"] >= 200
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print(f"wrote {args.output}")
    for family in output["families"]:
        counts = family["partition_counts"]
        print(
            f"{family['family_id']}: "
            f"seed={counts['T_seed_S0']} "
            f"val={counts['T_val_S0']} "
            f"test={counts['T_test_S0']} "
            f"train={counts['T_train_S0']}"
        )
    print(f"global counts: {output['global_partition_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
