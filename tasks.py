"""The concrete tasks: every default question, bound to the net_margin template."""

from env import env, net_margin
from quantlab_env.questions import default_questions

tasks = [net_margin(ticker=q.ticker, period_end=q.period_end, as_of=q.as_of) for q in default_questions()]

__all__ = ["env", "tasks"]
