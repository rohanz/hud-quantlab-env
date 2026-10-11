"""GRPO split `test_in` (grpo_splits.py, QUANTLAB_GRPO_SPLITS set): `hud ... --source tasks_grpo_test_in.py`."""

from grpo_splits import split_taskset
from tasks import env

_test_in = split_taskset("test_in")
tasks = list(_test_in)

__all__ = ["env", "tasks"]
