# Prompt: Matchup Research v2 (dynamic half, paired with matchup_research_system_v1.md)

**prompt_version:** `matchup_research_v2`
**Used by:** `src/nfl_predict/research/run_research.py` (Phase 6) as the Messages API user
turn; `src/nfl_predict/research/llm_provider.AnthropicMessagesProvider` supplies
`matchup_research_system_v1.md` separately as the cacheable `system` parameter (Phase 8A
cost-controls correction). Splitting the v1 template into a static system half and this
dynamic-only half changes NOTHING about research methodology, source rules, or output
requirements - see `matchup_research_system_v1.md`'s header for why this warranted a new
`prompt_version` anyway.
**Consumes:** a `ResearchInputPacket` (`src/nfl_predict/research/input_packet.py`) - the
quantitative model's already-computed predictions and the current market snapshot, never
influenced by this prompt.
**Produces:** a `submit_research_findings` tool call matching
`src/nfl_predict/research/schemas.py`'s `ResearchFindings` fields (see the system prompt's
"Required output format" section and `llm_provider._research_output_json_schema()`).

This is a template. Placeholders wrapped in double curly braces are filled in by
`run_research.build_prompt_from_template` at call time. Do not add a placeholder here
without also adding it to that function's substitution table - an unknown placeholder is a
hard error, not a silently-unfilled one. **Every placeholder here is genuinely game-specific
- never move one into the system file, or a cache hit would silently serve stale data for a
different game.**

---

## Context provided to you

- Matchup: `{{away_team_id}}` at `{{home_team_id}}`, `{{season}}` season, week `{{week}}`,
  kickoff `{{kickoff_timestamp}}`.
- Elo (frozen, independent model): predicted home margin `{{elo_predicted_margin}}`, home
  win probability `{{elo_home_win_probability}}`.
- Ridge margin (frozen, independent model): `{{ridge_predicted_margin}}` (may read
  `UNAVAILABLE` if no live feature pipeline has run for this season yet - do not treat that
  as a fact about the game).
- LightGBM (frozen, independent model): `{{lightgbm_predicted_margin}}` (same caveat).
- Market: home spread (traditional sign) `{{market_home_spread_traditional}}`, home
  moneyline `{{market_home_moneyline}}`, total `{{market_total_line}}` (may read
  `UNAVAILABLE`).
- Model-market disagreement, if computable: `{{model_market_disagreement_points}}` points
  (positive = model favors home more than the market does).

Research this matchup now, following the categories, source-tier rules, and output format
in your system instructions, then call `submit_research_findings` once with your complete
findings.
