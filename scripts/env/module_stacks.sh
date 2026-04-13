#!/usr/bin/env bash

_CAVER_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_ENV_DIR}/../common.sh"

caver_module_purge() {
  ensure_module_command
  module --force purge
}

activate_libero_stack() {
  caver_module_purge
  module load StdEnv/2020 python/3.8.10
}

activate_train_stack() {
  caver_module_purge
  module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 cudnn/9.10.0.56
}

activate_train_py310_stack() {
  caver_module_purge
  module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.10.13 cudnn/9.10.0.56
}

activate_gesim_stack() {
  activate_train_py310_stack
}

print_loaded_modules() {
  ensure_module_command
  module list 2>&1
}
