"""Point-in-time access to the SEC fixture, shipped as package data (quantlab_env/fixture.json)."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files


@cache
def load_fixture() -> dict:
    return json.loads(files(__package__).joinpath("fixture.json").read_text())


def companies() -> dict[str, dict]:
    return load_fixture()["companies"]


def concepts() -> list[str]:
    return list(load_fixture()["concepts"])


def facts_as_of(ticker: str, concept: str, as_of: str) -> list[dict]:
    """Annual facts as they were known on `as_of` (ISO date).

    For each fiscal period, the newest filing with `filed <= as_of` wins. A
    later restatement therefore replaces a period's value only from the day it
    was filed, and nothing filed after `as_of` is visible.
    """
    latest: dict[str, dict] = {}
    for row in companies()[ticker]["facts"][concept]:
        if row["filed"] > as_of:
            continue
        seen = latest.get(row["period_end"])
        if seen is None or (row["filed"], row["accn"]) > (seen["filed"], seen["accn"]):
            latest[row["period_end"]] = row
    return [latest[p] for p in sorted(latest)]
