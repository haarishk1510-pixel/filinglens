import type { NextConfig } from "next";

const demo = process.env.NEXT_PUBLIC_DEMO === "1";

const nextConfig: NextConfig = {
  // The public demo is a static site; the full app runs with its API.
  output: demo ? "export" : undefined,
  /* config options here */
  cacheComponents: !demo,
  partialPrefetching: !demo,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
