import type { MetadataRoute } from "next";
import { getAllGameIds } from "@/lib/api";

const BASE_URL = "https://www.edgemetricsports.com";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const staticRoutes: MetadataRoute.Sitemap = [
    { url: BASE_URL, changeFrequency: "daily", priority: 1 },
    { url: `${BASE_URL}/nfl/schedule`, changeFrequency: "daily", priority: 0.9 },
    { url: `${BASE_URL}/nfl/picks`, changeFrequency: "daily", priority: 0.9 },
    { url: `${BASE_URL}/nfl/expert`, changeFrequency: "daily", priority: 0.9 },
    { url: `${BASE_URL}/nfl/performance`, changeFrequency: "daily", priority: 0.7 },
    { url: `${BASE_URL}/methodology`, changeFrequency: "monthly", priority: 0.5 },
    { url: `${BASE_URL}/about`, changeFrequency: "monthly", priority: 0.5 },
  ];

  let gameRoutes: MetadataRoute.Sitemap = [];
  try {
    const { game_ids } = await getAllGameIds();
    gameRoutes = game_ids.map((game) => ({
      url: `${BASE_URL}/nfl/games/${game.game_id}`,
      changeFrequency: "daily" as const,
      priority: 0.6,
    }));
  } catch {
    // The real backend is unreachable at build time - ship the static routes rather than
    // failing the whole sitemap (and the build) over pages Google can still find via links.
  }

  return [...staticRoutes, ...gameRoutes];
}
