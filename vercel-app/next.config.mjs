/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    // Figures are PNGs served from /public; allow unoptimized for portable static export compatibility
    unoptimized: true,
  },
};

export default nextConfig;
