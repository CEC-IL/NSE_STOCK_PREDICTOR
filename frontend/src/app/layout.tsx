
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jenga | NSE Market Intelligence",
  description:
    "Explore Kenyan equities, historical market data and stock analysis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
