import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
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
  title: "APRR — Adaptive Probabilistic Routing Reinforcement",
  description:
    "Online policy-iteration routing for multi-agent LLM workflows. 35.7% latency reduction, 23.9% hop reduction, Pareto-optimal across baselines.",
  authors: [{ name: "Jenisha T" }],
  keywords: [
    "multi-agent LLM",
    "routing",
    "reinforcement learning",
    "online policy iteration",
    "APRR",
  ],
  openGraph: {
    title: "APRR — Adaptive Probabilistic Routing Reinforcement",
    description:
      "Online policy-iteration routing for multi-agent LLM workflows.",
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
