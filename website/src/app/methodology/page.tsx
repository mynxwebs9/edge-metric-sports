import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Methodology",
  description: "How our NFL prediction system works: independent and market-aware models, research, and a deterministic decision engine.",
};

const STEPS = [
  {
    title: "1. Quantitative models",
    body: "Every predicted score, win probability, and spread comes from versioned statistical models (Elo, Ridge regression, LightGBM) trained and backtested on historical NFL data. We keep an independent model, which never sees betting lines, structurally separate from a market-aware model, which can.",
  },
  {
    title: "2. Backtesting",
    body: "Before any model goes live, it's walk-forward backtested on sealed, previously-unseen seasons - never trained and tested on the same data. We publish that record honestly, including when the market beats our models.",
  },
  {
    title: "3. Market comparison",
    body: "We pull real sportsbook lines and compute a no-vig consensus across multiple books. The gap between our models and the market is one input to our decision engine, never something we chase on its own.",
  },
  {
    title: "4. Research",
    body: "A research step checks for information the structured data might miss - injuries, coaching changes, weather - and separates verified facts from unconfirmed reports and opinions. It never produces its own competing prediction or overrides a model number.",
  },
  {
    title: "5. Decision engine",
    body: "A deterministic, rules-based engine - not a human, not an LLM - combines the model, the market, and the research into one of five outcomes: No Bet, Watch, Lean, Best Bet, or Veto. Zero qualifying bets in a week is a valid, expected result.",
  },
  {
    title: "6. Verified tracking",
    body: "Every published pick is recorded in an immutable, append-only ledger the moment it's made - line, price, and reasoning included - and is never edited after the fact, win or lose.",
  },
];

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      <h1 className="mb-2 text-4xl font-black tracking-tight">Methodology</h1>
      <p className="mb-8 text-muted">
        A plain-language walkthrough of how a prediction on this site actually gets made - and what it doesn&apos;t mean.
      </p>
      <div className="space-y-6">
        {STEPS.map((step) => (
          <div key={step.title} className="rounded-xl border border-border bg-surface p-6 shadow-sm">
            <h2 className="mb-2 text-lg font-black tracking-tight">{step.title}</h2>
            <p className="text-sm text-muted">{step.body}</p>
          </div>
        ))}
      </div>
      <div className="mt-8 rounded-xl border border-border bg-surface-muted p-6 text-sm text-muted">
        A model favoring a team to win is not the same thing as a Best Bet. We only publish a pick when the model,
        the market, and current research all clear our thresholds together - see our{" "}
        <a href="/nfl/performance" className="font-semibold text-accent hover:underline">
          verified record
        </a>{" "}
        for exactly how that&apos;s worked out.
      </div>
    </div>
  );
}
