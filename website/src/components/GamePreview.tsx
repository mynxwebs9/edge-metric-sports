import type { PreviewBlock } from "@/lib/types";
import { formatTimestamp } from "@/lib/format";

export default function GamePreview({ preview }: { preview: PreviewBlock }) {
  if (!preview.available || !preview.text) return null;

  return (
    <section className="mb-8 rounded-xl border border-border bg-surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-xs font-bold uppercase tracking-wider text-muted">Game Preview</h2>
        <span className="rounded-full border border-border bg-surface-muted px-2 py-0.5 text-[11px] font-medium text-muted">
          AI-assisted analysis
        </span>
      </div>
      <p className="text-sm leading-relaxed text-foreground">{preview.text}</p>
      <p className="mt-3 text-xs text-muted">Updated {formatTimestamp(preview.generated_at)}</p>
    </section>
  );
}
