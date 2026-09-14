"""Phase 5 Step 10: does larger model-vs-market spread disagreement correspond to better
ATS results? Buckets `ats.ATSBetRecord`s by their (already-fixed, predetermined)
`edge_bucket` and reports each bucket's W-L-P/cover-rate/ROI, split by season - "2024
positive AND 2025 positive" is more convincing than one season carrying an aggregate, per
the brief's explicit instruction not to hide a losing season inside a profitable aggregate.
"""

from __future__ import annotations

from nfl_predict.market.ats import ATSBetRecord, summarize_ats_bets
from nfl_predict.market.disagreement import EDGE_BUCKET_LABELS


def spread_edge_bucket_report(bets: list[ATSBetRecord]) -> dict:
    report: dict = {}
    for bucket in EDGE_BUCKET_LABELS:
        bucket_bets = [b for b in bets if b.edge_bucket == bucket]
        report[bucket] = {
            "combined_2024_2025": summarize_ats_bets(bucket_bets),
            "season_2024": summarize_ats_bets([b for b in bucket_bets if b.season == 2024]),
            "season_2025": summarize_ats_bets([b for b in bucket_bets if b.season == 2025]),
        }
    report["all_bets_any_disagreement"] = {
        "combined_2024_2025": summarize_ats_bets(bets),
        "season_2024": summarize_ats_bets([b for b in bets if b.season == 2024]),
        "season_2025": summarize_ats_bets([b for b in bets if b.season == 2025]),
    }
    return report
