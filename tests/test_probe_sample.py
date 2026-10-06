"""The frozen probe subsets (v1 and v2): seeded, stratified by question type, drawn from the probe sets."""

import hashlib
from collections import Counter

from quantlab_env.questions import (
    CHAIN_KINDS,
    ONE_HOP_KINDS,
    PROBE_SAMPLE_PER_KIND,
    PROBE_V2_COUNTS,
    chain_questions,
    probe_questions,
    probe_sample,
    probe_sample_v2,
)


def test_subset_has_twenty_of_each_kind_and_comes_from_the_probe_set():
    full = probe_questions()
    sample = probe_sample(full)
    assert Counter(q.kind for q in sample) == {k: PROBE_SAMPLE_PER_KIND for k in {q.kind for q in full}}
    assert {q.prompt for q in sample} <= {q.prompt for q in full}
    assert len({q.prompt for q in sample}) == len(sample)


def test_subset_does_not_depend_on_input_order():
    full = probe_questions()
    assert [q.prompt for q in probe_sample(full)] == [q.prompt for q in probe_sample(full[::-1])]


def test_subset_is_frozen():
    # Pinned before any model ran on the probe (2026-10-06). A change here means the fixture,
    # the probe generators or the sampler changed, and every model must be rerun on the new subset.
    digest = hashlib.sha256("\n".join(q.prompt for q in probe_sample(probe_questions())).encode()).hexdigest()
    assert digest == "f41b9cdbae8ff3c0246f2b1c426376cbf5463e3291dfe31ec6fb5590a07c69c8"


def test_v2_subset_is_weighted_to_the_chained_types_and_drawn_from_both_sets():
    full = probe_questions() + chain_questions()
    sample = probe_sample_v2(full)
    assert Counter(q.kind for q in sample) == PROBE_V2_COUNTS and len(sample) == 100
    assert all(PROBE_V2_COUNTS[k] == 20 for k in CHAIN_KINDS) and all(PROBE_V2_COUNTS[k] == 4 for k in ONE_HOP_KINDS)
    assert {q.prompt for q in sample} <= {q.prompt for q in full}
    assert len({q.prompt for q in sample}) == len(sample)
    assert [q.prompt for q in sample] == [q.prompt for q in probe_sample_v2(full[::-1])]


def test_v2_subset_is_frozen():
    # Pinned before any model ran on it (2026-10-06). tasks_probe100.py (v1) is pinned above and unchanged.
    digest = hashlib.sha256(
        "\n".join(q.prompt for q in probe_sample_v2(probe_questions() + chain_questions())).encode()
    ).hexdigest()
    assert digest == "9cde6035723745581a635ab06e572421de55f8de6054b73a208a15d37ad0503e"
