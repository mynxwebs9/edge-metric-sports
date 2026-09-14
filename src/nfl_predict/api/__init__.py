"""FastAPI read API: the only path the website uses to reach storage. Implemented starting Phase 8.

The website never imports src/nfl_predict directly and never opens the database itself — it
only calls the JSON endpoints this package exposes. See docs/ARCHITECTURE.md#website-api-boundary
and docs/WEBSITE_SPEC.md.
"""
