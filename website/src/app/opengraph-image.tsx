import { ImageResponse } from "next/og";

export const alt = "Edge Metric Sports - NFL Predictions";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: "80px",
          background: "#13181a",
          color: "#ffffff",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 36 }}>
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 14,
              background: "#0f766e",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 32,
              fontWeight: 900,
            }}
          >
            E
          </div>
          <div style={{ fontSize: 30, fontWeight: 700, letterSpacing: 4, textTransform: "uppercase", color: "#9fb0b0" }}>
            Edge Metric Sports
          </div>
        </div>
        <div style={{ display: "flex", fontSize: 68, fontWeight: 900, letterSpacing: -1, lineHeight: 1.1 }}>
          Real NFL predictions.
        </div>
        <div style={{ display: "flex", fontSize: 68, fontWeight: 900, letterSpacing: -1, lineHeight: 1.1, color: "#2dd4bf" }}>
          Verified results.
        </div>
        <div style={{ display: "flex", fontSize: 28, color: "#9fb0b0", marginTop: 28 }}>
          A quantitative model, real market lines, and a public track record - win or lose.
        </div>
      </div>
    ),
    { ...size },
  );
}
