import Link from "next/link";
import { Logo } from "./Logo";

const LINKS = [
  { href: "#results", label: "Results" },
  { href: "#charts", label: "Charts" },
  { href: "#figures", label: "Figures" },
  { href: "#algorithm", label: "Algorithm" },
  { href: "#reproduce", label: "Reproduce" },
];

export function Nav() {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-ink-900/80 backdrop-blur-md">
      <nav className="section-pad flex h-14 items-center justify-between">
        <Link href="#top" className="flex items-center gap-2.5 text-fg">
          <Logo className="h-6 w-6" />
          <span className="text-sm font-semibold tracking-tight">APRR</span>
          <span className="hidden text-xs text-fg-dim sm:inline">
            Adaptive Probabilistic Routing Reinforcement
          </span>
        </Link>
        <div className="flex items-center gap-1">
          <div className="hidden items-center gap-1 md:flex">
            {LINKS.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className="rounded-md px-3 py-1.5 text-sm text-fg-muted transition-colors hover:bg-ink-700 hover:text-fg"
              >
                {l.label}
              </a>
            ))}
          </div>
          <a
            href="https://github.com/joyjeni/aprr-multi-agent-routing"
            target="_blank"
            rel="noreferrer"
            className="ml-1 rounded-md border border-line px-3 py-1.5 text-sm font-medium text-fg transition-colors hover:border-aprr hover:text-aprr-soft"
          >
            GitHub
          </a>
        </div>
      </nav>
    </header>
  );
}
