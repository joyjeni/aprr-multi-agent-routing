import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "APRR · ACRS objective O3: risk-weighted policy routing",
  description:
    "Per-step routing between Qwen3-4B and Qwen3-8B inside ACRS, evaluated on AppWorld: 21.8% of steps on the 4B model and 49% fewer 8B calls at unchanged task success (development split, one seed). Earlier ToolBench simulation results included.",
  authors: [{ name: "Jenisha T" }],
  keywords: [
    "multi-agent LLM",
    "routing",
    "reinforcement learning",
    "online policy iteration",
    "APRR",
    "AppWorld",
    "ACRS",
  ],
  openGraph: {
    title: "APRR · ACRS objective O3: risk-weighted policy routing",
    description:
      "Risk-weighted routing between a small and a large open model, evaluated on AppWorld.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body className="bg-ink-900 text-fg antialiased">{children}</body>
    </html>
  );
}
