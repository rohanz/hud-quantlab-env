"""The frozen 100-task probe subset (20 per question type, seeded): `hud ... --source tasks_probe100.py`.

Drawn by `questions.probe_sample` from the full probe set before any model ran on it, so the
subset does not depend on model results. Its slugs are pinned in tests/test_probe_sample.py.
"""

from hud.eval import Taskset

from quantlab_env.questions import probe_questions, probe_sample
from tasks import as_tasks, env

_probe100 = Taskset("quantlab-probe100", as_tasks(probe_sample(probe_questions())))
tasks = list(_probe100)

__all__ = ["env", "tasks"]
