#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_exe="/home/sevecl/Desktop/PY/Kinter/.zmq_env/bin/python"
exec "$python_exe" "$script_dir/main.py" "$@"
