"""Known values in the checked-in fixture, read straight from the JSON.

Independent of quantlab_env (no facts_as_of, no gold computation): the figures
below were checked by hand against the companies' 10-K income statements.
"""

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "src" / "quantlab_env" / "fixture.json"


def value(ticker: str, concept: str, period_end: str, accn: str) -> int:
    rows = json.loads(FIXTURE.read_text())["companies"][ticker]["facts"][concept]
    (row,) = [r for r in rows if r["period_end"] == period_end and r["accn"] == accn]
    return row["value"]


@pytest.mark.parametrize(
    ("ticker", "concept", "period_end", "accn", "expected"),
    [
        # JNJ FY2021 as first reported (10-K filed 2022-02-17) and restated after the
        # Kenvue separation (10-K filed 2024-02-16). Net income was not restated.
        ("JNJ", "revenue", "2022-01-02", "0000200406-22-000022", 93_775_000_000),
        ("JNJ", "revenue", "2022-01-02", "0000200406-24-000013", 78_740_000_000),
        ("JNJ", "net_income", "2022-01-02", "0000200406-22-000022", 20_878_000_000),
        ("JNJ", "net_income", "2022-01-02", "0000200406-24-000013", 20_878_000_000),
        # AAPL FY2023 (10-K filed 2023-11-03).
        ("AAPL", "revenue", "2023-09-30", "0000320193-23-000106", 383_285_000_000),
        ("AAPL", "net_income", "2023-09-30", "0000320193-23-000106", 96_995_000_000),
        # KO FY2023 (10-K filed 2024-02-20).
        ("KO", "revenue", "2023-12-31", "0000021344-24-000009", 45_754_000_000),
        ("KO", "net_income", "2023-12-31", "0000021344-24-000009", 10_714_000_000),
    ],
)
def test_known_values(ticker: str, concept: str, period_end: str, accn: str, expected: int):
    assert value(ticker, concept, period_end, accn) == expected


def test_restated_jnj_fy2021_margin():
    # The restatement trap's answer, by hand: 20,878 / 78,740 = 26.52%.
    margin = 20_878_000_000 / value("JNJ", "revenue", "2022-01-02", "0000200406-24-000013")
    assert round(margin * 100, 2) == 26.52
