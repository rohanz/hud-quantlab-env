"""The legacy set alone: `hud ... --source tasks_legacy.py`."""

from tasks import env
from tasks import legacy as _legacy

tasks = list(_legacy)

__all__ = ["env", "tasks"]
