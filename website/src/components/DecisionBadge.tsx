const DECISION_STYLES: Record<string, string> = {
  QUALIFIED_BET: "bg-qualified/15 text-qualified border-qualified/30",
  LEAN: "bg-lean/15 text-lean border-lean/30",
  WATCH: "bg-watch/15 text-watch border-watch/30",
  VETO: "bg-veto/15 text-veto border-veto/30",
  NO_BET: "bg-no-bet/15 text-no-bet border-no-bet/30",
};

export default function DecisionBadge({ decision, label }: { decision: string | null; label: string | null }) {
  if (!decision || !label) {
    return (
      <span className="inline-flex items-center rounded-full border border-border bg-surface-muted px-3 py-1 text-xs font-semibold text-muted">
        Pending
      </span>
    );
  }
  const styles = DECISION_STYLES[decision] ?? "bg-surface-muted text-muted border-border";
  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${styles}`}>
      {label}
    </span>
  );
}
