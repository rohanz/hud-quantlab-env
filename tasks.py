"""The concrete tasks, as two named sets bound to the env's templates.

legacy  the 18 net-margin questions that name the period by its end date
        (`questions.default_questions`)
probe   the harder probe set, derived from the fixture (`questions.probe_questions`)

`hud ... --source tasks.py` sees both sets. To run one, use tasks_legacy.py or
tasks_probe.py.
"""

from hud.eval import Taskset

from env import TEMPLATES, env
from quantlab_env.questions import Question, default_questions, probe_questions, template_args


def as_tasks(qs: list[Question]) -> list:
    return [TEMPLATES[q.kind](**template_args(q)) for q in qs]


legacy = Taskset("quantlab-legacy", as_tasks(default_questions()))
probe = Taskset("quantlab-probe", as_tasks(probe_questions()))

__all__ = ["env", "legacy", "probe"]
