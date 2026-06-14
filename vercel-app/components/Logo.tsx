export function Logo({ className = "h-7 w-7" }: { className?: string }) {
  // Abstract routing mark: three nodes, a reinforced (red) edge selected
  // among baseline (muted) edges. Geometric, works at 24px and 200px.
  return (
    <svg
      className={className}
      viewBox="0 0 32 32"
      fill="none"
      aria-label="APRR routing mark"
      role="img"
    >
      {/* baseline edges */}
      <path d="M6 24 L16 7" stroke="currentColor" strokeOpacity="0.28" strokeWidth="1.6" />
      <path d="M6 24 L26 24" stroke="currentColor" strokeOpacity="0.28" strokeWidth="1.6" />
      {/* reinforced edge */}
      <path d="M16 7 L26 24" stroke="#d93025" strokeWidth="2.4" strokeLinecap="round" />
      {/* nodes */}
      <circle cx="6" cy="24" r="3" fill="currentColor" />
      <circle cx="16" cy="7" r="3.4" fill="#d93025" />
      <circle cx="26" cy="24" r="3" fill="currentColor" />
    </svg>
  );
}
