"""Phase 4 Steps 2-3 orchestrator: runs the walk-forward protocol and immediately persists
every prediction to the immutable ledger, before any outcome is ever joined.

    python -m nfl_predict.backtesting.run_backtest

Requires the Phase 4 freeze to already be complete (`nfl_predict.backtesting.run_freeze`).
"""

from __future__ import annotations

import json

from nfl_predict.logging_conf import get_logger

from nfl_predict.backtesting.ledger import write_prediction_ledger
from nfl_predict.backtesting.walk_forward import run_walk_forward

logger = get_logger(__name__)

RUN_ID = "phase4_v1"


def run(run_id: str = RUN_ID) -> dict:
    logger.info("Starting Phase 4 walk-forward backtest, run_id=%s", run_id)
    records = run_walk_forward()
    logger.info("Walk-forward produced %d prediction records; writing to the immutable ledger", len(records))
    result = write_prediction_ledger(records, run_id)
    logger.info("Ledger written to %s (sha256=%s, n=%d)", result.ledger_path, result.ledger_sha256, result.n_predictions)
    return {
        "run_id": run_id,
        "ledger_path": str(result.ledger_path),
        "hash_path": str(result.hash_path),
        "ledger_sha256": result.ledger_sha256,
        "n_predictions": result.n_predictions,
    }


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
