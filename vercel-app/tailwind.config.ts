import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#0a0a0b",
          900: "#0a0a0b",
          800: "#121214",
          700: "#1b1b1f",
          600: "#26262c",
          500: "#3a3a42",
        },
        // Accent (blue) for this work; neutral slate for links and baselines; amber for the oracle.
        // No red or green anywhere on the site.
        aprr: {
          DEFAULT: "#256abf",
          soft: "#6da7ec",
          dim: "#1c5cab",
        },
        base: {
          DEFAULT: "#7c8aa5",
          soft: "#a5b4cf",
          dim: "#2c3444",
        },
        oracle: "#c98500",
        line: "#26262c",
        fg: {
          DEFAULT: "#f5f5f7",
          muted: "#a1a1aa",
          dim: "#6b6b73",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      maxWidth: {
        content: "1180px",
      },
    },
  },
  plugins: [],
};

export default config;
