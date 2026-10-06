"""The frozen probe v2 subset (100 tasks, seeded): `hud ... --source tasks_probe_v2.py`.

20 of each chained analyst question type and 4 of each v1 probe type, drawn by
`questions.probe_sample_v2` from the probe and chained sets before any model ran on them, so
the subset does not depend on model results. Its prompts are pinned in
tests/test_probe_sample.py. tasks_probe100.py (v1) is unchanged.
"""

from hud.eval import Taskset

from quantlab_env.questions import chain_questions, probe_questions, probe_sample_v2
from tasks import as_tasks, env

_probe_v2 = Taskset("quantlab-probe-v2", as_tasks(probe_sample_v2(probe_questions() + chain_questions())))
tasks = list(_probe_v2)

__all__ = ["env", "tasks"]
