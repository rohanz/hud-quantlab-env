"""GRPO split `test_heldout` (grpo_splits.py, QUANTLAB_GRPO_SPLITS set): `hud ... --source tasks_grpo_test_heldout.py`."""

from grpo_splits import split_taskset
from tasks import env

_test_heldout = split_taskset("test_heldout")
tasks = list(_test_heldout)

__all__ = ["env", "tasks"]
