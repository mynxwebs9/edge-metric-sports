import type { PerformanceResponse } from "@/lib/types";

export default function PerformanceHeadline({ performance }: { performance: PerformanceResponse }) {
  if (performance.headline) {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-qualified/30 bg-qualified/10 px-4 py-1.5 text-sm font-semibold text-qualified">
        🔥 {performance.headline.toUpperCase()}
      </div>
    );
  }
  return (
    <p className="text-sm text-muted">
      {performance.has_settled_history
        ? "Verified performance tracking is underway - check back as more picks settle."
        : "No settled Best Bets yet this season - every pick is tracked from publication, win or lose."}
    </p>
  );
}
