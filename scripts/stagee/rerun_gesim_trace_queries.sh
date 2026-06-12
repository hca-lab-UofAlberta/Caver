#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  rerun_gesim_trace_queries.sh --trace-path PATH --context-id ID --query-indices IDS [options]

Required:
  --trace-path PATH          Stage-E chunk trace JSONL path
  --context-id ID            Context id inside the trace
  --query-indices IDS        Comma-separated 1-based policy query indices

Options:
  --output-root PATH         Root directory for rerun artifacts
  --prompt TEXT              Prompt string passed to GE-Sim inference
  --dry-run                  Print the resolved commands without executing
  -h, --help                 Show this message
EOF
}

trace_path=""
context_id=""
query_indices=""
output_root=""
prompt="best quality, consistent and smooth motion, realistic, clear and distinct."
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --trace-path)
      trace_path="${2:?missing value for --trace-path}"
      shift 2
      ;;
    --context-id)
      context_id="${2:?missing value for --context-id}"
      shift 2
      ;;
    --query-indices)
      query_indices="${2:?missing value for --query-indices}"
      shift 2
      ;;
    --output-root)
      output_root="${2:?missing value for --output-root}"
      shift 2
      ;;
    --prompt)
      prompt="${2:?missing value for --prompt}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: ${1}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command python3

if [ -z "${trace_path}" ] || [ -z "${context_id}" ] || [ -z "${query_indices}" ]; then
  usage >&2
  exit 1
fi

trace_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${trace_path}")"
if [ ! -f "${trace_path}" ]; then
  echo "error: trace path not found: ${trace_path}" >&2
  exit 1
fi

if [ -z "${output_root}" ]; then
  output_root="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/gesim_trace_query_reruns__$(timestamp_utc)"
fi
ensure_directory "${output_root}"
output_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${output_root}")"

query_list="$(
  python3 - "${query_indices}" <<'PY'
import sys
values = []
for token in sys.argv[1].split(","):
    token = token.strip()
    if not token:
        continue
    value = int(token)
    if value <= 0:
        raise SystemExit(f"invalid query index: {value}")
    values.append(value)
if not values:
    raise SystemExit("no query indices provided")
print("\n".join(str(v) for v in values))
PY
)"

queries_manifest_path="${output_root}/queries_manifest.jsonl"
: > "${queries_manifest_path}"

for query_index in ${query_list}; do
  query_info="$(
    python3 - "${trace_path}" "${context_id}" "${query_index}" <<'PY'
import json
import pathlib
import sys

trace_path = pathlib.Path(sys.argv[1])
context_id = sys.argv[2]
query_index = int(sys.argv[3])

with trace_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if context_id not in line:
            continue
        record = json.loads(line)
        if record.get("context_id") != context_id:
            continue
        if int(record.get("policy_query_index", -1)) != query_index:
            continue
        metadata_path = pathlib.Path(record["selected_provider_aux"]["metadata_path"]).resolve()
        bundle_dir = metadata_path.parent
        payload = {
            "query_index": query_index,
            "selected_candidate_index": record.get("selector", {}).get("selected_candidate_index"),
            "metadata_path": str(metadata_path),
            "bundle_dir": str(bundle_dir),
        }
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0)

raise SystemExit(
    f"error: could not find context_id={context_id!r} query_index={query_index} in {trace_path}"
)
PY
  )"
  printf '%s\n' "${query_info}" >> "${queries_manifest_path}"

  bundle_dir="$(
    python3 -c 'import json,sys; print(json.loads(sys.argv[1])["bundle_dir"])' "${query_info}"
  )"
  query_tag="$(printf 'query_%03d' "${query_index}")"
  query_root="${output_root}/${query_tag}"
  gesim_output_dir="${query_root}/gesim_rerun"
  ensure_directory "${query_root}"

  rerun_cmd=(
    "${CAVER_REPO_ROOT}/scripts/stagee/run_gesim_inference.sh"
    --bundle-dir "${bundle_dir}"
    --output-dir "${gesim_output_dir}"
    --prompt "${prompt}"
  )
  printf 'query %s rerun command:' "${query_index}"
  printf ' %q' "${rerun_cmd[@]}"
  printf '\n'
  if (( ! dry_run )); then
    "${rerun_cmd[@]}"
  fi

  motion_summary_path="${query_root}/motion_summary.json"
  motion_cmd=(
    "${CAVER_REPO_ROOT}/scripts/env/with_gesim_infer.sh"
    --
    python
    -
    "${gesim_output_dir}"
    "${motion_summary_path}"
  )
  if (( dry_run )); then
    printf 'query %s motion command:' "${query_index}"
    printf ' %q' "${motion_cmd[@]}"
    printf '\n'
  else
    "${motion_cmd[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

output_dir = Path(sys.argv[1]).resolve()
summary_path = Path(sys.argv[2]).resolve()
raw = torch.load(output_dir / "video.pt", map_location="cpu")
if hasattr(raw, "detach"):
    raw = raw.detach().cpu().numpy()
runtime = yaml.safe_load((output_dir / "gesim_runtime.yaml").read_text(encoding="utf-8"))
frame_height, frame_width = [int(value) for value in runtime["data"]["train"]["sample_size"]]
valid_cam = [str(value) for value in runtime["data"]["train"]["valid_cam"]]

payload = {
    "frame_count": int(raw.shape[0]),
    "frame_height": frame_height,
    "frame_width": frame_width,
    "valid_cam": valid_cam,
    "motion": {},
}
for view_index, camera_name in enumerate(valid_cam):
    crop = raw[:, :frame_height, view_index * frame_width : (view_index + 1) * frame_width, :].astype("float32")
    diffs = []
    for step in range(len(crop) - 1):
        diffs.append(float(abs(crop[step + 1] - crop[step]).mean()))
    payload["motion"][camera_name] = {
        "frame_mean_absdiff": diffs,
        "mean": float(sum(diffs) / len(diffs)) if diffs else 0.0,
    }

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
  fi
done

printf 'query manifest: %s\n' "${queries_manifest_path}"
printf 'rerun root: %s\n' "${output_root}"
