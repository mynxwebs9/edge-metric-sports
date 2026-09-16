// The ONLY place this app talks to the backend - every page/component fetches data through
// these functions, never a raw fetch() scattered through component code, and never a direct
// database/model/provider call (see docs/ARCHITECTURE.md#website-api-boundary). No other
// file in this app may import a Python module, open a database connection, or call
// Anthropic/The Odds API - there is nothing here that could even attempt it, since this is a
// plain HTTP client against the FastAPI service's JSON endpoints.

import type {
  BestBetsResponse,
  DecisionsResponse,
  GameDetail,
  GameIdsResponse,
  ModelStatusResponse,
  PerformanceResponse,
  ScheduleWeeksResponse,
  SlateResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string, revalidateSeconds: number): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, { next: { revalidate: revalidateSeconds } });
  if (!res.ok) {
    throw new ApiError(res.status, `${path} returned HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// Short revalidation windows - real data changes as new pipeline runs land, and this app
// never caches stale predictions as if they were current (see docs/WEBSITE_SPEC.md).
export function getCurrentSlate(): Promise<SlateResponse> {
  return getJson<SlateResponse>("/api/nfl/slate/current", 30);
}

export function getGameDetail(gameId: string): Promise<GameDetail> {
  return getJson<GameDetail>(`/api/nfl/games/${encodeURIComponent(gameId)}`, 30);
}

export function getBestBets(): Promise<BestBetsResponse> {
  return getJson<BestBetsResponse>("/api/nfl/best-bets", 30);
}

export function getPerformance(): Promise<PerformanceResponse> {
  return getJson<PerformanceResponse>("/api/nfl/performance", 60);
}

export function getModelStatus(): Promise<ModelStatusResponse> {
  return getJson<ModelStatusResponse>("/api/nfl/model-status", 60);
}

export function getDecisions(gameId: string): Promise<DecisionsResponse> {
  return getJson<DecisionsResponse>(`/api/nfl/decisions/${encodeURIComponent(gameId)}`, 30);
}

export function getScheduleWeeks(): Promise<ScheduleWeeksResponse> {
  return getJson<ScheduleWeeksResponse>("/api/nfl/schedule", 60);
}

export function getScheduleForWeek(week: number): Promise<SlateResponse> {
  return getJson<SlateResponse>(`/api/nfl/schedule/${week}`, 30);
}

// Deliberately cheap - just real game_ids for the whole season, none of the per-game
// model/market/research/decision data `getScheduleForWeek` fetches. Built for the sitemap,
// which needs every game's URL but not its data - fetching all of that per week was slow
// enough to time out sitemap generation in production.
export function getAllGameIds(): Promise<GameIdsResponse> {
  return getJson<GameIdsResponse>("/api/nfl/game-ids", 300);
}
