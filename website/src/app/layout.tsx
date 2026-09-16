import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const SITE_TITLE = "Edge Metric Sports — NFL Predictions";
const SITE_DESCRIPTION = "Data-driven NFL predictions, market comparisons, and a verified betting record - free and ad-supported.";

export const metadata: Metadata = {
  metadataBase: new URL("https://edgemetricsports.com"),
  title: {
    default: SITE_TITLE,
    template: "%s | Edge Metric Sports",
  },
  description: SITE_DESCRIPTION,
  openGraph: {
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    url: "https://edgemetricsports.com",
    siteName: "Edge Metric Sports",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        {/* Plain literal <script> tag, deliberately not next/script - AdSense's site-ownership
            verification checks for this exact tag's presence in the raw page source, and
            next/script's "beforeInteractive" strategy renders only a <link rel="preload"> plus
            a JS-executed bootstrap call in production, never a literal <script src="..."> - a
            known mismatch with AdSense's snippet verification. Confirmed by inspecting the
            actual prerendered production HTML output. */}
        <script
          async
          src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-1516960995259341"
          crossOrigin="anonymous"
        />
      </head>
      <body className="flex min-h-full flex-col">
        <SiteHeader />
        <main className="flex-1">{children}</main>
        <SiteFooter />
        <Analytics />
      </body>
    </html>
  );
}
