import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Marginal — paper reader",
  description: "See every summary point connected to the passage it came from.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
