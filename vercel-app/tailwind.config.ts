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
        aprr: {
          DEFAULT: "#d93025",
          soft: "#ef5350",
          dim: "#7a1d18",
        },
        base: {
          DEFAULT: "#1a73e8",
          soft: "#4f93f0",
          dim: "#123a6e",
        },
        oracle: "#1e9e6a",
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
