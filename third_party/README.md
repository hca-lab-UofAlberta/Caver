# Third-Party Layout

This directory is the SDRE-local home for external checkouts and Python environments used by the CAVER study.

Expected untracked layout:

- `third_party/src/`
  - source checkouts for `openpi`, `LIBERO`, `pi-StepNFT`, and optional GE-Sim / RLinf paths
- `third_party/venvs/`
  - component-specific or shared Python virtual environments
- `third_party/wheelhouse/`
  - optional cached wheels if later needed for repeated installs

The bootstrap scripts in `scripts/env/` are written against this layout by default.

