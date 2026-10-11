"""GRPO split `train` (grpo_splits.py, QUANTLAB_GRPO_SPLITS set): `hud ... --source tasks_grpo_train.py`."""

from grpo_splits import split_taskset
from tasks import env

_train = split_taskset("train")
tasks = list(_train)

__all__ = ["env", "tasks"]
