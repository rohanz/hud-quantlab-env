"""The GRPO splits as Tasksets: rows written by quantlab-generality's scripts/hud_split.py.

Each `tasks_grpo_<split>.py` module poses one split's rows (`{"args", "kind", "slug"}`
JSON lines) through the env's own templates, as tasks.py does for the named sets. The rows
live in the training repo (pinned by sha256 there), so the directory holding them comes
from `QUANTLAB_GRPO_SPLITS`. Without it the split modules raise ImportError, so a
directory scan of this repo skips them and a direct `--source tasks_grpo_<split>.py`
fails loudly. A row whose slug the template no longer reproduces fails too: the env
changed since the split was written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from hud.eval import Taskset

from env import TEMPLATES

SPLITS_ENV = "QUANTLAB_GRPO_SPLITS"


def split_taskset(name: str) -> Taskset:
    root = os.environ.get(SPLITS_ENV)
    if not root:
        raise ImportError(f"set {SPLITS_ENV} to the directory holding {name}.jsonl (scripts/hud_split.py)")
    lines = (Path(root) / f"{name}.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    tasks = [TEMPLATES[row["kind"]](**row["args"]) for row in rows]
    changed = [row["slug"] for row, task in zip(rows, tasks, strict=True) if task.slug != row["slug"]]
    if changed:
        raise ValueError(f"{name}.jsonl: the env no longer poses {len(changed)} rows as written, e.g. {changed[0]}")
    return Taskset(f"quantlab-grpo-{name}", tasks)
