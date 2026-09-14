import type { ReasonCodeOut } from "@/lib/types";

export default function ReasonCodeList({ codes }: { codes: ReasonCodeOut[] }) {
  if (codes.length === 0) {
    return <p className="text-sm text-muted">No reasons recorded yet.</p>;
  }
  return (
    <ul className="space-y-1.5">
      {codes.map((code) => (
        <li key={code.code} className="flex items-start gap-2 text-sm text-foreground">
          <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
          {code.label}
        </li>
      ))}
    </ul>
  );
}
