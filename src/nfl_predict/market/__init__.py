"""Sportsbook market data and market-benchmark analysis (Phase 5+). Kept structurally
separate from `src/nfl_predict/models` and `src/nfl_predict/features` per CLAUDE.md
principle 2 - the independent football model never imports from this package, and nothing
in this package feeds back into `nfl_predict.models` or `nfl_predict.backtesting`'s frozen
Phase 3/4 artifacts. See docs/PHASE5_MARKET_REPORT.md.
"""
