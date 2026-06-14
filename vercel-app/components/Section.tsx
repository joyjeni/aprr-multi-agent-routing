export function Section({
  id,
  eyebrow,
  title,
  intro,
  children,
}: {
  id: string;
  eyebrow: string;
  title: string;
  intro?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="section-pad scroll-mt-20 py-14 sm:py-20">
      <div className="max-w-2xl">
        <span className="eyebrow">{eyebrow}</span>
        <h2 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">
          {title}
        </h2>
        {intro && (
          <p className="mt-3 text-base leading-relaxed text-fg-muted">{intro}</p>
        )}
      </div>
      <div className="mt-8">{children}</div>
    </section>
  );
}
