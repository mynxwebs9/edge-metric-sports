// Phase 8B, item 14: reserved ad space, no ad network integrated yet. A real ad network
// swaps this component's contents in a later phase - nothing else in the app should need to
// change when that happens, since every placement already renders this one component.

export default function AdSlot({ label, className = "" }: { label: string; className?: string }) {
  return (
    <div
      className={`flex items-center justify-center rounded-lg border border-dashed border-border bg-surface-muted text-xs text-muted ${className}`}
      style={{ minHeight: "90px" }}
      data-ad-slot={label}
    >
      Ad space reserved · {label}
    </div>
  );
}
