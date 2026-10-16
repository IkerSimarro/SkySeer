#!/usr/bin/env bash
# Set up a SkySeer station on a Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit).
#
#   git clone https://github.com/IkerSimarro/SkySeer.git ~/skyseer
#   cd ~/skyseer && ./deploy/pi/install.sh [--gps]
#
# --gps also sets up gpsd and chrony, so a USB GPS gives the station its location and
# keeps the clock right without internet (see deploy/pi/README.md).
set -euo pipefail

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
USER_NAME="$(id -un)"
WITH_GPS=0
[[ "${1:-}" == "--gps" ]] && WITH_GPS=1

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

step "System packages (camera library, Python, video tools)"
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-venv python3-dev git
if (( WITH_GPS )); then
  sudo apt-get install -y gpsd gpsd-clients chrony pps-tools
fi

step "Python environment"
# --system-site-packages lets the environment see picamera2/libcamera, which come from
# the operating system and can't be installed with pip.
cd "$DIR"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

step "Dashboard"
if [[ -f src/skyseer/api/static/index.html ]]; then
  echo "Dashboard already built."
elif command -v npm >/dev/null && [[ "$(node -p 'process.versions.node.split(".")[0]')" -ge 20 ]]; then
  (cd web && npm ci && npm run build)
else
  echo "Node.js 20+ not found: build the dashboard on your computer and copy it over:"
  echo "  (cd web && npm ci && npm run build)"
  echo "  scp -r src/skyseer/api/static ${USER_NAME}@$(hostname).local:${DIR}/src/skyseer/api/"
fi

step "Star database and satellite orbits (first time: several minutes)"
.venv/bin/python - <<'PY'
from skyseer.astro.catalog import build_database
from skyseer.config import load_config
a = load_config("configs/pi.yaml").astro
build_database(a.database_min_fov, a.database_max_fov, a.database_max_mag, a.data_dir)
print("Star database ready")
PY
.venv/bin/skyseer orbits --update || echo "No internet: orbits will download later."

if (( WITH_GPS )); then
  step "GPS: gpsd and chrony"
  sudo tee /etc/default/gpsd >/dev/null <<'GPSD'
# SkySeer: a USB GPS receiver; -n keeps it tracking even with no client connected.
DEVICES="/dev/ttyACM0"
GPSD_OPTIONS="-n"
USBAUTO="true"
GPSD
  sudo tee /etc/chrony/conf.d/skyseer-gps.conf >/dev/null <<'CHRONY'
# SkySeer: take the time from the GPS when there is no internet.
# NMEA gives the time to ~0.1 s; a PPS wire (if the receiver has one) to microseconds.
refclock SHM 0 refid GPS precision 1e-1 offset 0.2 delay 0.2
# refclock PPS /dev/pps0 lock GPS refid PPS   # uncomment if PPS is wired
makestep 1 3
CHRONY
  sudo systemctl restart gpsd chrony
fi

step "Station service (starts at boot)"
sed -e "s|@USER@|${USER_NAME}|g" -e "s|@DIR@|${DIR}|g" deploy/pi/skyseer.service \
  | sudo tee /etc/systemd/system/skyseer.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now skyseer

step "Done"
echo "Dashboard: http://$(hostname).local:8000  (or this Pi's IP address)"
echo "Log:       journalctl -u skyseer -f"
echo "Location:  set observer: in configs/pi.yaml, or use a GPS (--gps)."
