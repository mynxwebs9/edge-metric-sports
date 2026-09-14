"""Phase 7 Step 26: runs the decision engine against the existing Phase 6 prospective pilot
packets (still pregame-valid - kickoff has not passed) and persists the resulting
decisions. Does NOT modify the Phase 6 research packets in any way. Does NOT force a
QUALIFIED_BET - if the engine returns NO_BET/WATCH/LEAN/VETO, that is recorded as-is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from nfl_predict.logging_conf import get_logger

from nfl_predict.decision.decision_log import append_decision_record
from nfl_predict.decision.engine import decide
from nfl_predict.decision.input_packet import build_decision_packet_from_research_run
from nfl_predict.decision.rules_config import load_rule_set
from nfl_predict.decision.schemas import DecisionRecord

logger = get_logger(__name__)

SEASON = 2026
WEEK = 1
RESEARCH_RUN_ID = "pilot_20260911"
DECISION_TIMESTAMP = "2026-09-11T21:00:00+00:00"
GAMES = ["2026_01_DEN_KC", "2026_01_BUF_HOU", "2026_01_CHI_CAR"]


def main() -> int:
    rule_set = load_rule_set()
    results = {}

    for game_id in GAMES:
        packet = build_decision_packet_from_research_run(SEASON, WEEK, game_id, RESEARCH_RUN_ID, decision_timestamp=DECISION_TIMESTAMP)
        packet_hash = hashlib.sha256(json.dumps(asdict(packet), sort_keys=True, default=str).encode("utf-8")).hexdigest()

        game_results = {}
        for market_type in ("spread", "moneyline"):
            decision, reasons = decide(packet, rule_set, market_type)
            record = DecisionRecord(
                decision_id=f"{game_id}_{market_type}_{RESEARCH_RUN_ID}", game_id=game_id,
                decision_timestamp=DECISION_TIMESTAMP, kickoff_timestamp=packet.kickoff_timestamp,
                decision=decision, reason_codes=reasons, decision_rule_version=rule_set.rule_set_version,
                market_type=market_type, input_packet_hash=packet_hash, validation_status="PROSPECTIVE",
            )
            append_decision_record(record, SEASON, WEEK)
            game_results[market_type] = {"decision": decision.value, "reason_codes": [r.value for r in reasons]}
            logger.info("Decision for %s/%s: %s %s", game_id, market_type, decision.value, [r.value for r in reasons])

        results[game_id] = game_results

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
