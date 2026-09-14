"""Season-range parsing shared by the CLI and coverage reporting.

Keeps the historical window (2010-2025 for this project's initial foundation) out of code
as a spec: callers pass a spec string, never a hardcoded list of years scattered through the
codebase. See docs/PHASE1_DATA_REPORT.md for the window actually used to build the
foundation and CLAUDE.md/docs/ARCHITECTURE.md for why it's configurable.
"""

from __future__ import annotations

# Not a hardcoded assumption baked into ingestion logic — just the CLI's default value when
# --seasons is omitted, so a plain "ingest everything" run has a sane default without
# forcing every invocation to spell out the window. Any range is usable via --seasons.
DEFAULT_HISTORICAL_SEASONS = "2010-2025"


def parse_season_range(spec: str) -> list[int]:
    """Parse a season-range spec into a sorted, de-duplicated list of season years.

    Accepted forms:
      - a single year: "2025"
      - a hyphenated inclusive range: "2010-2025"
      - a comma-separated list: "2023,2024,2025"
      - a comma-separated list of ranges: "2010-2015,2020,2023-2025"
    """
    if not spec or not spec.strip():
        raise ValueError("Season range spec must not be empty.")

    years: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, _, end_str = chunk.partition("-")
            start, end = _parse_year(start_str), _parse_year(end_str)
            if end < start:
                raise ValueError(f"Invalid range '{chunk}': end year before start year.")
            years.update(range(start, end + 1))
        else:
            years.add(_parse_year(chunk))

    if not years:
        raise ValueError(f"Season range spec '{spec}' did not resolve to any seasons.")
    return sorted(years)


def _parse_year(value: str) -> int:
    value = value.strip()
    if not value.isdigit() or len(value) != 4:
        raise ValueError(f"'{value}' is not a valid 4-digit season year.")
    year = int(value)
    if not (1920 <= year <= 2100):
        raise ValueError(f"Season year {year} is outside a plausible NFL range.")
    return year
