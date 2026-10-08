"""Fireball alerts: tell someone when a very bright meteor is seen.

Alerts are always written to the console log and, by default, to ``data/alerts.jsonl``.
Network notifications are opt-in: a generic JSON webhook and/or a push notification via
the free ntfy app (https://ntfy.sh). Sending happens in a background thread with a short
timeout so a slow network never delays detection.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any

from .config import AlertConfig, PhotometryConfig
from .models import Event, ObjectClass

log = logging.getLogger(__name__)


def is_fireball(event: Event, config: PhotometryConfig) -> bool:
    p = event.photometry
    return (
        event.label is ObjectClass.METEOR
        and p is not None
        and p.peak_magnitude is not None
        and p.peak_magnitude <= config.fireball_magnitude
    )


class Alerter:
    def __init__(self, config: AlertConfig, photometry: PhotometryConfig):
        self.config = config
        self.photometry = photometry
        self.sent: list[dict[str, Any]] = []

    def check(self, event: Event) -> bool:
        """Raise an alert if the event is a fireball. Returns True if it was."""
        if not is_fireball(event, self.photometry):
            return False
        event.fireball = True
        self._send(self._payload(event))
        return True

    @staticmethod
    def _payload(event: Event) -> dict[str, Any]:
        p = event.photometry
        shower = event.shower.short_label() if event.shower is not None else "sporadic"
        where = ""
        if event.sky is not None and event.sky.alt is not None and event.sky.az is not None:
            where = f", alt {event.sky.alt[0]:.0f}° az {event.sky.az[0]:.0f}°"
        return {
            "type": "fireball",
            "time": event.track.start_time.isoformat(),
            "event_id": event.id,
            "peak_magnitude": None if p is None else p.peak_magnitude,
            "saturated": None if p is None else p.saturated,
            "shower": shower,
            "duration_s": round(event.track.duration, 2),
            "message": f"Fireball! magnitude {p.peak_label if p else '?'} ({shower}) at "
            f"{event.track.start_time:%H:%M:%S} UTC{where}",
        }

    def _send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)
        log.warning(payload["message"])
        c = self.config
        if c.log_file:
            path = Path(c.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        if c.webhook_url or c.ntfy_topic:
            threading.Thread(target=self._post, args=(payload,), daemon=True).start()

    def _post(self, payload: dict[str, Any]) -> None:
        c = self.config
        try:
            if c.webhook_url:
                req = urllib.request.Request(
                    c.webhook_url,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=10).close()
            if c.ntfy_topic:
                req = urllib.request.Request(
                    f"https://ntfy.sh/{c.ntfy_topic}",
                    data=payload["message"].encode(),
                    headers={"Title": "SkySeer fireball", "Priority": "high", "Tags": "comet"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=10).close()
        except OSError as e:  # network problems must never stop the station
            log.warning("Could not deliver fireball alert: %s", e)
