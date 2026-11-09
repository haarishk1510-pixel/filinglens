import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FilingLens",
  description: "Ask any company filing. Get answers you can verify.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
