#!/usr/bin/env bash
# SkySeer's Python environment on Raspberry Pi OS (run by install.sh, and by CI).
#
#   ./deploy/pi/setup-python.sh [extra pip arguments, e.g. --group dev]
#
# --system-site-packages lets the environment see picamera2/libcamera, which come from
# the operating system and can't be installed with pip. The camera library's JPEG code
# (simplejpeg) is compiled against the system's numpy, so numpy stays on the system's
# major version: numpy 1.x on bookworm, 2.x on trixie. A newer release of the same major
# version is fine; numpy 2 in front of a numpy 1 build breaks `import picamera2`.
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
major="$(python3 -c 'import numpy; print(numpy.__version__.split(".")[0])')"
.venv/bin/pip install -e . "numpy==${major}.*" "$@"
.venv/bin/python -c "import picamera2" && echo "Camera library loads: OK"
