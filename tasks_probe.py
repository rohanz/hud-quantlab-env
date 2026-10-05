"""The probe set alone: `hud ... --source tasks_probe.py`."""

from tasks import env
from tasks import probe as _probe

tasks = list(_probe)

__all__ = ["env", "tasks"]
