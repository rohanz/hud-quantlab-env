"""GRPO split `val` (grpo_splits.py, QUANTLAB_GRPO_SPLITS set): `hud ... --source tasks_grpo_val.py`."""

from grpo_splits import split_taskset
from tasks import env

_val = split_taskset("val")
tasks = list(_val)

__all__ = ["env", "tasks"]
