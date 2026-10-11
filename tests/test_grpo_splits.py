"""The GRPO split modules (grpo_splits.py): rows written elsewhere, posed by the env's templates."""

import importlib
import json
import sys

import pytest

import grpo_splits
from env import TEMPLATES

TASK = TEMPLATES["net_margin"](ticker="JNJ", period_end="2022-01-02", as_of="2024-03-17")


def _write(tmp_path, name, slug=TASK.slug):
    row = {"args": TASK.args, "kind": "net_margin", "slug": slug}
    (tmp_path / f"{name}.jsonl").write_text(json.dumps(row) + "\n")


def _fresh(module):
    sys.modules.pop(module, None)
    return importlib.import_module(module)


def test_rows_become_tasks_through_the_templates(tmp_path, monkeypatch):
    _write(tmp_path, "val")
    monkeypatch.setenv(grpo_splits.SPLITS_ENV, str(tmp_path))
    assert [t.slug for t in grpo_splits.split_taskset("val")] == [TASK.slug]
    assert [t.slug for t in _fresh("tasks_grpo_val").tasks] == [TASK.slug]


def test_a_row_the_env_no_longer_poses_fails(tmp_path, monkeypatch):
    _write(tmp_path, "train", slug="net_margin-00000000")
    monkeypatch.setenv(grpo_splits.SPLITS_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="no longer poses"):
        grpo_splits.split_taskset("train")


def test_without_the_variable_the_split_modules_raise_import_error(monkeypatch):
    """So a directory scan of the repo skips them, and a direct --source fails loudly."""
    monkeypatch.delenv(grpo_splits.SPLITS_ENV, raising=False)
    for name in ("train", "val", "test_in", "test_heldout"):
        with pytest.raises(ImportError, match=grpo_splits.SPLITS_ENV):
            _fresh(f"tasks_grpo_{name}")
