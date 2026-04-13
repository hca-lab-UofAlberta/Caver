#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


STAGE0_PARTITIONS = ("T_seed_S0", "T_train_S0", "T_val_S0", "T_test_S0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced Stage-0 family manifest slice for Stage-E experiment runs."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--partition-name", default="T_train_S0", choices=STAGE0_PARTITIONS)
    parser.add_argument(
        "--family-ids",
        default="",
        help="Optional comma-separated family ids. Defaults to all families in manifest order.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Total number of contexts to select across the chosen families.",
    )
    parser.add_argument(
        "--family-offset",
        type=int,
        default=0,
        help="Starting offset into each selected family's partition before taking contexts.",
    )
    parser.add_argument(
        "--round-size",
        type=int,
        default=25,
        help="Stage-E budget round size for metadata only.",
    )
    return parser.parse_args()


def parse_family_ids(raw_family_ids: str) -> list[str]:
    return [value.strip() for value in raw_family_ids.split(",") if value.strip()]


def main() -> int:
    args = parse_args()

    if args.budget < 1:
        raise SystemExit("error: --budget must be positive")
    if args.family_offset < 0:
        raise SystemExit("error: --family-offset must be non-negative")
    if args.round_size < 1:
        raise SystemExit("error: --round-size must be positive")

    input_manifest_path = Path(args.input_manifest).resolve()
    output_manifest_path = Path(args.output_manifest).resolve()

    with input_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)

    requested_family_ids = parse_family_ids(args.family_ids)
    requested_family_id_set = set(requested_family_ids)

    selected_source_families = []
    selected_family_ids = []
    for family in source_manifest["families"]:
        family_id = family["family_id"]
        if requested_family_id_set and family_id not in requested_family_id_set:
            continue
        selected_source_families.append(family)
        selected_family_ids.append(family_id)

    if requested_family_id_set:
        missing_family_ids = sorted(requested_family_id_set.difference(selected_family_ids))
        if missing_family_ids:
            raise SystemExit(f"error: requested family ids were not found in manifest: {missing_family_ids}")

    if not selected_source_families:
        raise SystemExit("error: selection produced zero families")

    family_count = len(selected_source_families)
    if args.budget % family_count != 0:
        raise SystemExit(
            f"error: --budget {args.budget} is not divisible by the selected family count {family_count}"
        )
    contexts_per_family = args.budget // family_count
    if contexts_per_family < 1:
        raise SystemExit("error: balanced selection would assign zero contexts per family")

    derived_families = []
    backend_member_suites = set()
    backend_member_tasks = []

    for family in selected_source_families:
        family_partition_contexts = list(family["partitions"][args.partition_name])
        end_offset = args.family_offset + contexts_per_family
        if end_offset > len(family_partition_contexts):
            raise SystemExit(
                "error: family "
                f"{family['family_id']} has only {len(family_partition_contexts)} contexts in {args.partition_name}, "
                f"cannot satisfy offset {args.family_offset} plus {contexts_per_family} contexts"
            )

        selected_contexts = family_partition_contexts[args.family_offset:end_offset]
        partition_counts = {partition_name: 0 for partition_name in STAGE0_PARTITIONS}
        partition_counts[args.partition_name] = len(selected_contexts)
        partitions = {partition_name: [] for partition_name in STAGE0_PARTITIONS}
        partitions[args.partition_name] = selected_contexts

        member_tasks = list(family.get("member_tasks", []))
        for member in member_tasks:
            backend_member_suites.add(member["suite"])
            backend_member_tasks.append(int(member["task_index"]))

        derived_families.append(
            {
                "family_id": family["family_id"],
                "proposal_task": family.get("proposal_task"),
                "proxy_label": family.get("proxy_label"),
                "substitution_note": family.get("substitution_note"),
                "member_tasks": member_tasks,
                "available_contexts": family.get("available_contexts"),
                "source_partition_available_contexts": len(family_partition_contexts),
                "partition_counts": partition_counts,
                "partitions": partitions,
            }
        )

    output_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/stagee/build_stage0_balanced_manifest.py",
        "source_manifest": str(input_manifest_path),
        "notes": [
            "Derived balanced Stage-E manifest slice.",
            "Each selected family contributes the same number of contexts from the chosen source partition.",
        ],
        "selection": {
            "partition_name": args.partition_name,
            "requested_family_ids": requested_family_ids,
            "selected_family_ids": selected_family_ids,
            "family_count": family_count,
            "budget": args.budget,
            "contexts_per_family": contexts_per_family,
            "family_offset": args.family_offset,
            "round_size": args.round_size,
        },
        "backend": {
            "task_suite_names": sorted(backend_member_suites),
            "task_ids": sorted(set(backend_member_tasks)),
        },
        "global_partition_counts": {
            partition_name: (args.budget if partition_name == args.partition_name else 0)
            for partition_name in STAGE0_PARTITIONS
        },
        "families": derived_families,
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
