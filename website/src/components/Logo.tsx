// The mark: a line hugging a baseline, then breaking sharply upward - the visual idea of
// "finding an edge" (the model's read breaking away from the market's baseline), which is
// the actual product this site is built around, not a generic abstract icon.

export default function Logo({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <svg width="34" height="34" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" className="shrink-0">
        <rect width="32" height="32" rx="8" className="fill-accent" />
        <line x1="6" y1="21" x2="26" y2="21" stroke="white" strokeOpacity="0.35" strokeWidth="1.5" strokeDasharray="2 2.5" />
        <path d="M7 22 L13 19.5 L17.5 22.5 L22 11 L26 7.5" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        <circle cx="26" cy="7.5" r="2.1" fill="white" />
      </svg>
      <div className="leading-none">
        <div className="text-[17px] font-black uppercase tracking-tight text-foreground">Edge Metric</div>
        <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-accent">Sports</div>
      </div>
    </div>
  );
}
