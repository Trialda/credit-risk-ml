import type { ExplainResponse } from "../api/client";

interface Props {
  result: ExplainResponse;
}

export default function RiskResult({ result }: Props) {
  const percentage = (result.risk_score * 100).toFixed(1);
  const risk = result.risk_score > 0.5 ? "High" : "Low";
  const color = result.risk_score > 0.5 ? "red" : "green";

  const topFeatures = result.shap_features.slice(0, 8);
  const maxAbsShap = Math.max(
    ...topFeatures.map((f) => Math.abs(f.shap_value))
  );

  return (
    <div style={{ marginTop: "2rem" }}>
      <h2>Risk Assessment</h2>

      <p>
        <strong>Score:</strong> {percentage}%
      </p>
      <p>
        <strong>Risk level:</strong>{" "}
        <span style={{ color }}>{risk}</span>
      </p>
      <p style={{ fontSize: "0.8rem", color: "gray" }}>
        Base value: {(result.base_value * 100).toFixed(1)}%
      </p>

      <h3>Top contributing factors</h3>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {topFeatures.map((feature) => {
          const isPositive = feature.shap_value > 0;
          const barWidth =
            maxAbsShap > 0
              ? (Math.abs(feature.shap_value) / maxAbsShap) * 100
              : 0;

          return (
            <div key={feature.feature}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.85rem",
                  marginBottom: "2px",
                }}
              >
                <span>{feature.feature}</span>
                <span style={{ color: isPositive ? "red" : "green" }}>
                  {isPositive ? "+" : ""}
                  {feature.shap_value.toFixed(4)}
                </span>
              </div>
              <div
                style={{
                  height: "8px",
                  backgroundColor: "#eee",
                  borderRadius: "4px",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${barWidth}%`,
                    backgroundColor: isPositive ? "#d62728" : "#1f77b4",
                    borderRadius: "4px",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p style={{ fontSize: "0.75rem", color: "gray", marginTop: "1rem" }}>
        Request ID: {result.request_id}
      </p>
    </div>
  );
}