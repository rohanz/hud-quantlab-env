"""The frozen probe subset: seeded, stratified by question type, a subset of the probe set."""

import hashlib
from collections import Counter

from quantlab_env.questions import PROBE_SAMPLE_PER_KIND, probe_questions, probe_sample


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
