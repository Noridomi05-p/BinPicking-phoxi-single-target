#!/usr/bin/env bash
set -euo pipefail
calibration_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$calibration_project_dir"
if [[ ! -x .venv/bin/python ]]; then
  echo "Chưa có .venv/bin/python trong thư mục dự án. Tạo / kích hoạt môi trường Python của dự án trước." >&2
  exit 1
fi
exec .venv/bin/python scripts/16_calibrate_gui.py "$@"
