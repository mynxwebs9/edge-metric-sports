"""Phase 6 Step 25: the real prospective pilot - three real, current, upcoming NFL games
(Week 1, 2026 season), researched using real WebSearch results gathered in this session on
2026-09-11, run through the actual research pipeline via `ManualResearchProvider` (see
`src/nfl_predict/research/manual_provider.py` for why this is not an automated API call).

Run once: `python scripts/run_phase6_pilot.py`. Each game's research JSON below is authored
directly from real search results (see docs/PHASE6_RESEARCH_AGENT_REPORT.md for the exact
queries and sources) - every source_url is real, every fact is exactly what was found, no
embellishment.
"""

from __future__ import annotations

import json

from nfl_predict.logging_conf import get_logger
from nfl_predict.research.input_packet import build_live_input_packet
from nfl_predict.research.manual_provider import ManualResearchProvider
from nfl_predict.research.run_research import run_research_for_game

logger = get_logger(__name__)

RESEARCH_TIMESTAMP = "2026-09-11T20:30:00+00:00"

# ---------------------------------------------------------------------------
# Game 1: DEN @ KC (home=KC) - MNF, 2026-09-14. LARGE Elo/market disagreement.
# ---------------------------------------------------------------------------
DEN_KC_FINDINGS = {
    "material_facts": [
        {
            "text": "Chiefs OT Josh Simmons (back) did not participate in practice and is not expected to play Monday night.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://www.chiefs.com/news/week-1-injury-report-broncos-vs-chiefs-2026", "source_title": "Week 1 Injury Report | Broncos vs. Chiefs", "publisher": "KansasCityChiefs.com (official team site)", "source_tier": "TIER_1_OFFICIAL", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 2, "confidence_in_fact": 0.9,
            "reason": "Starting-caliber offensive tackle out is a real but not season-altering loss; MODERATE not MAJOR since backup depth exists and this is one lineman, not a unit-wide issue.",
        },
        {
            "text": "Chiefs S Chamarri Conner (knee) did not participate in practice and is not expected to play Monday night.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://www.chiefs.com/news/week-1-injury-report-broncos-vs-chiefs-2026", "source_title": "Week 1 Injury Report | Broncos vs. Chiefs", "publisher": "KansasCityChiefs.com (official team site)", "source_tier": "TIER_1_OFFICIAL", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 1, "confidence_in_fact": 0.9,
            "reason": "A single safety out is a depth-chart-level change, MINOR for overall team strength.",
        },
        {
            "text": "QB Patrick Mahomes is returning from a torn ACL; he had full practice participation this week and Chiefs HC Andy Reid said there is 'a good chance' Mahomes starts Monday night.",
            "category": "VERIFIED_FACT",
            "sources": [
                {"source_url": "https://www.yardbarker.com/nfl/articles/patrick_mahomes_headlines_chiefs_eight_man_injury_report_for_week_1_broncos_clash/s1_17664_44288136", "source_title": "Patrick Mahomes headlines Chiefs' eight-man injury report for week 1 Broncos clash", "publisher": "Yardbarker", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None},
                {"source_url": "https://www.chiefs.com/news/week-1-injury-report-broncos-vs-chiefs-2026", "source_title": "Week 1 Injury Report | Broncos vs. Chiefs", "publisher": "KansasCityChiefs.com (official team site)", "source_tier": "TIER_1_OFFICIAL", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None},
            ],
            "materiality_level": 3, "confidence_in_fact": 0.85,
            "reason": "MAJOR, not CRITICAL: full practice participation is a strong positive signal he will play close to normally, but a torn-ACL return still carries real uncertainty about performance level that a coach's hedged 'good chance' phrasing does not fully resolve.",
        },
        {
            "text": "Denver Broncos: all 53 roster players participated in Thursday practice - the team is at full strength entering the game.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://www.si.com/nfl/broncos/onsi/broncos-chiefs-first-injury-report-striking-week-1-contrast-simmons-mims", "source_title": "Broncos-Chiefs First Injury Report Reveals a Striking Week 1 Contrast", "publisher": "Sports Illustrated (Mile High Huddle)", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 1, "confidence_in_fact": 0.85,
            "reason": "A fully healthy roster is a mild positive for Denver but not itself surprising or extreme - MINOR.",
        },
    ],
    "uncertain_reports": [],
    "external_model_opinions": [],
    "analyst_opinions": [
        {
            "text": "Multiple betting-market writeups describe this as a 'striking contrast' in Week 1 health between a banged-up Chiefs roster and a fully healthy Broncos roster.",
            "category": "ANALYST_OPINION",
            "sources": [{"source_url": "https://www.si.com/nfl/broncos/onsi/broncos-chiefs-first-injury-report-striking-week-1-contrast-simmons-mims", "source_title": "Broncos-Chiefs First Injury Report Reveals a Striking Week 1 Contrast", "publisher": "Sports Illustrated", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 1, "confidence_in_fact": 0.7, "reason": "Framing/opinion about the health gap, not a new fact beyond what's already listed above.",
        },
    ],
    "qb_status": "Chiefs starter Patrick Mahomes (returning from a torn ACL) had full practice participation this week; HC Andy Reid says there is a 'good chance' he starts Monday night. No other QB status changes found for either team.",
    "ol_status": "Chiefs OT Josh Simmons (back) is out. No offensive-line issues reported for Denver.",
    "skill_position_status": "No material skill-position injuries reported for either team.",
    "defensive_personnel_status": "Chiefs S Chamarri Conner (knee) is out - a single depth-chart-level loss, not a unit-wide issue. Denver reported at full strength.",
    "weather_status": "Not checked - indoor/outdoor status and forecast were not part of this research pass.",
    "coaching_status": "No coordinator/scheme changes reported for either team.",
    "missing_information": [
        "No live market-snapshot object was available through this pipeline for this game (Phase 5's live odds provider is unconfigured) - the spread/moneyline/total referenced in the input packet's market section is UNAVAILABLE by that path; a current line WAS found via web search (Chiefs -2.5 opening, moved to -3, total 42.5) and is noted here for context only, not as a structured market field.",
        "No official confirmation (as of this research pass) that Mahomes is 100% cleared for full game action versus a managed workload plan.",
        "Weather/roof status not checked.",
    ],
    "research_classification": "SUPPORTS_MARKET",
}
# Rationale for SUPPORTS_MARKET, recorded here (not in the stored JSON) for the pilot
# report: Elo (continued forward from the frozen 2010-2025 ratings) favors Denver by ~3.2
# points, but every fact found in this research points toward Kansas City being reasonably
# healthy at the two positions that matter most (Mahomes practicing fully; only a swing
# tackle and a depth safety out) while Denver, though fully healthy, has no specific
# positive news beyond "healthy." The market's ~2.5-3 point favorite lean toward Kansas
# City looks better-supported by CURRENT information than Elo's team-strength prior, which
# cannot see this week's practice-participation reports at all.

# ---------------------------------------------------------------------------
# Game 2: BUF @ HOU (home=HOU) - 2026-09-13 early window. MODERATE disagreement.
# ---------------------------------------------------------------------------
BUF_HOU_FINDINGS = {
    "material_facts": [
        {
            "text": "Bills RB Ty Johnson (hamstring) did not participate in practice; he has a notable pass-protection role, with veteran Ray Davis in line for more reps if he can't play.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://www.buffalorumblings.com/buffalo-bills-analysis/139375/braden-smiths-injury-a-big-one-for-texans-before-buffalo-bills-game-in-week-1", "source_title": "Braden Smith's injury a big one for Texans before Buffalo Bills game in Week 1", "publisher": "Buffalo Rumblings", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 1, "confidence_in_fact": 0.85,
            "reason": "A backup-role RB's pass-protection snaps are a real but MINOR consideration, not a starter-level loss.",
        },
        {
            "text": "Bills DB Jordan Hancock (quad) did not participate in practice.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://www.buffalorumblings.com/buffalo-bills-injuries/139650/buffalo-bills-at-houston-texans-injury-tracker", "source_title": "See which Bills and Texans were listed on the first injury report for Week 1", "publisher": "Buffalo Rumblings", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 0, "confidence_in_fact": 0.8,
            "reason": "A depth defensive back's absence is NOISE-level for team strength.",
        },
        {
            "text": "Starting quarterbacks for both teams (Josh Allen for Buffalo, C.J. Stroud for Houston) are confirmed and do not appear on either team's injury report.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://sports.yahoo.com/articles/bills-vs-texans-injury-buffalo-131744894.html", "source_title": "Bills vs. Texans injury update: Buffalo has 5 key players dealing with injuries", "publisher": "Yahoo Sports", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 0, "confidence_in_fact": 0.9,
            "reason": "Confirms no QB-driven surprise is in play for either side - NOISE (informational, rules out a major risk category rather than introducing one).",
        },
    ],
    "uncertain_reports": [],
    "external_model_opinions": [
        {"source": "Public betting split (via Buffalo Rumblings' Yahoo Sports aggregation)", "prediction_text": "Buffalo is taking 70% of early spread tickets and 72% of early spread money; moneyline play leans further toward Buffalo at 78% of bets / 75% of money.", "market": "spread", "line_at_publication": "BUF -1.5", "publication_timestamp": None, "stated_confidence": None},
    ],
    "analyst_opinions": [],
    "qb_status": "Both starters (Allen, Stroud) confirmed, no injury concern found for either.",
    "ol_status": "No Bills or Texans offensive-line starter absences found. (An initial search suggested a Texans OT injury headline; on inspection the actual current injury report showed Texans OT Trent Brown as a full practice participant, not out - included here to be transparent about a headline that did not hold up on closer reading.)",
    "skill_position_status": "Bills RB Ty Johnson (hamstring, pass-protection role) is out; no other skill-position injuries of note.",
    "defensive_personnel_status": "Bills DB Jordan Hancock (quad) out; Texans DT Kayden McDonald (ankle) limited. Neither is a starter-level unit concern.",
    "weather_status": "Not checked - dome/outdoor status not part of this research pass.",
    "coaching_status": "No coordinator/scheme changes reported for either team.",
    "missing_information": [
        "No live market-snapshot object was available through this pipeline (Phase 5's live odds provider is unconfigured); a current line was found via web search (BUF -1.5, ML BUF -120/HOU +100, total 44.5) and is noted for context only.",
        "Weather/roof status not checked.",
    ],
    "research_classification": "MIXED",
}
# Rationale for MIXED: Elo favors the home team (Houston) by 2 points; the current market
# favors the road team (Buffalo) by 1.5 - a real, moderate disagreement (roughly 3.5
# points) that this research does not resolve. Nothing found here (minor injuries on both
# sides, confirmed healthy starting QBs) clearly explains or supports either side's read.

# ---------------------------------------------------------------------------
# Game 3: CHI @ CAR (home=CAR) - 2026-09-13 early window. SMALL disagreement, direction agrees.
# ---------------------------------------------------------------------------
CHI_CAR_FINDINGS = {
    "material_facts": [
        {
            "text": "Bears starting LT Ozzy Trapilo (knee) is questionable and is NOT expected to play against Carolina.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://sports.yahoo.com/articles/bears-vs-panthers-final-injury-193858394.html", "source_title": "Bears vs Panthers Final Injury Report: 4 Questionable for Chicago", "publisher": "Yahoo Sports", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 3, "confidence_in_fact": 0.8,
            "reason": "A starting left tackle sitting out on the road is a MAJOR offensive-line concern, per the Step 4 research-question rubric's own framing of OL starter availability.",
        },
        {
            "text": "Bears backup QB Tyson Bagent (back) is questionable - starter Caleb Williams is not on the injury report and is expected to start normally.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://sports.yahoo.com/articles/bears-vs-panthers-final-injury-193858394.html", "source_title": "Bears vs Panthers Final Injury Report: 4 Questionable for Chicago", "publisher": "Yahoo Sports", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 0, "confidence_in_fact": 0.85,
            "reason": "The backup QB's status is NOISE for game expectations since the starter is confirmed healthy - included explicitly so it is not mistaken for a starting-QB concern.",
        },
        {
            "text": "Bears WR Rome Odunze (calf) is questionable but trending toward playing.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://bvmsports.com/2026/09/10/panthers-vs-bears-week-1-full-preview-injury-report-depth-charts-predictions-more/", "source_title": "Panthers vs Bears Week 1 FULL Preview: Injury Report, Depth Charts, Predictions & More", "publisher": "BVM Sports", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": "2026-09-10T00:00:00+00:00"}],
            "materiality_level": 1, "confidence_in_fact": 0.7,
            "reason": "A questionable-but-trending-toward-playing WR is MINOR - the expected outcome is he plays.",
        },
        {
            "text": "Bears S Xavier Woods (groin) is questionable.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://bvmsports.com/2026/09/10/panthers-vs-bears-week-1-full-preview-injury-report-depth-charts-predictions-more/", "source_title": "Panthers vs Bears Week 1 FULL Preview: Injury Report, Depth Charts, Predictions & More", "publisher": "BVM Sports", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": "2026-09-10T00:00:00+00:00"}],
            "materiality_level": 1, "confidence_in_fact": 0.7, "reason": "A single questionable safety is MINOR.",
        },
        {
            "text": "Panthers OLB Patrick Jones II missed practice again (pass-rush specialist).",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://theleadsm.com/how-the-panthers-can-pull-off-the-week-one-upset/", "source_title": "Panthers Week One: Game Preview and Insights", "publisher": "The Lead", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 2, "confidence_in_fact": 0.7,
            "reason": "A missed-practice pass rusher, twice, is MODERATE - plausibly affects Carolina's ability to pressure Chicago's line, but is one situational piece, not season-defining.",
        },
        {
            "text": "Panthers RB Jonathon Brooks returned to practice in a limited capacity.",
            "category": "VERIFIED_FACT",
            "sources": [{"source_url": "https://theleadsm.com/how-the-panthers-can-pull-off-the-week-one-upset/", "source_title": "Panthers Week One: Game Preview and Insights", "publisher": "The Lead", "source_tier": "TIER_2_REPUTABLE_REPORTER", "retrieval_timestamp": RESEARCH_TIMESTAMP, "publication_timestamp": None}],
            "materiality_level": 1, "confidence_in_fact": 0.7, "reason": "A positive recovery signal for a Carolina skill player - MINOR.",
        },
    ],
    "uncertain_reports": [],
    "external_model_opinions": [],
    "analyst_opinions": [],
    "qb_status": "Both starters (Caleb Williams for Chicago, Bryce Young for Carolina) are healthy and expected to start. Chicago's backup QB Tyson Bagent is questionable, which is not material given the starter's status.",
    "ol_status": "Bears starting LT Ozzy Trapilo (knee) is out - the single most material fact found in this research pass for this game.",
    "skill_position_status": "Bears WR Rome Odunze (calf, trending toward playing); Panthers RB Jonathon Brooks (limited practice, positive trend). Neither is expected to be a game-changing absence.",
    "defensive_personnel_status": "Bears S Xavier Woods (groin) questionable; Panthers OLB Patrick Jones II has missed multiple practices (pass rush). Both are situational, not unit-wide.",
    "weather_status": "Not checked - dome/outdoor status not part of this research pass.",
    "coaching_status": "No coordinator/scheme changes reported for either team.",
    "missing_information": [
        "No live market-snapshot object was available through this pipeline (Phase 5's live odds provider is unconfigured); a current line was found via web search (Bears -2.5 on the road) and is noted for context only.",
        "Weather/roof status not checked.",
        "No confirmation of Trapilo's replacement's specific recent snap count/continuity.",
    ],
    "research_classification": "MIXED",
}
# Rationale for MIXED: Elo (favoring Chicago by ~1.5) and the current market (favoring
# Chicago by ~2.5) already agree in direction, with only a modest magnitude gap between
# them - not a large disagreement. The most material fact found (Chicago's starting LT
# out) is a real negative for the favored team's offensive line, which is context for why
# neither side is picking Chicago by a wide margin, but does not clearly push the
# disagreement toward either side being "more right" - hence MIXED rather than
# SUPPORTS_MODEL/SUPPORTS_MARKET.

GAMES = [
    ("2026_01_DEN_KC", DEN_KC_FINDINGS),
    ("2026_01_BUF_HOU", BUF_HOU_FINDINGS),
    ("2026_01_CHI_CAR", CHI_CAR_FINDINGS),
]


def main() -> int:
    prompt_template = open("prompts/matchup_research.md", encoding="utf-8").read()
    results = {}
    for game_id, findings_dict in GAMES:
        packet = build_live_input_packet(game_id)
        provider = ManualResearchProvider(research_output_text=json.dumps(findings_dict))
        run_id = "pilot_20260911"
        result = run_research_for_game(
            packet=packet, provider=provider, research_prompt_template_text=prompt_template,
            run_id=run_id, now=RESEARCH_TIMESTAMP,
        )
        results[game_id] = result
        logger.info("Pilot research complete for %s: %s", game_id, result)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
