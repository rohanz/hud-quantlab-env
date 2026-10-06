"""The concrete tasks, as three named sets bound to the env's templates.

legacy  the 18 net-margin questions that name the period by its end date
        (`questions.default_questions`)
probe   the harder one-hop probe set, derived from the fixture (`questions.probe_questions`)
chains  the chained analyst questions: peers ranked, a company against its peers, a margin
        trend (`questions.chain_questions`)

`hud ... --source tasks.py` sees all three sets. To run one, use tasks_legacy.py or
tasks_probe.py; the frozen subsets are tasks_probe100.py (v1) and tasks_probe_v2.py.
"""

from hud.eval import Taskset

from env import TEMPLATES, env
from quantlab_env.questions import Question, chain_questions, default_questions, probe_questions, template_args


def as_tasks(qs: list[Question]) -> list:
    return [TEMPLATES[q.kind](**template_args(q)) for q in qs]


legacy = Taskset("quantlab-legacy", as_tasks(default_questions()))
probe = Taskset("quantlab-probe", as_tasks(probe_questions()))
chains = Taskset("quantlab-chains", as_tasks(chain_questions()))

__all__ = ["chains", "env", "legacy", "probe"]
