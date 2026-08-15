import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["pdfjs-dist"],
  experimental: { serverActions: { bodySizeLimit: "20mb" } },
};

export default nextConfig;
