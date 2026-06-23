import type { ExplainResponse } from "../api/client";

interface Props {
  result: ExplainResponse;
}

export default function RiskResult({ result }: Props) {
  // Threshold chosen from evaluate.py's threshold analysis (see MLflow run
  // 95e2d9f26bea48fdb1d1b701e21c864e), NOT the naive midpoint of 0.5, which
  // given an 8% base default rate barely filters the applicant pool at all
  // (default_rate_at_0_5 = 4.8%, only marginally better than the unconditional
  // base rate of 8.07%). 0.2 targets a meaningfully safer approval pool
  // (default_rate_at_0_2 = 2.25%) while keeping the approval rate ~27%.
  const score = result.risk_score;
  const percentage = (score * 100).toFixed(1);

  let risk: "Low" | "Medium" | "High";
  let color: "green" | "orange" | "red";
  let decision: "Approve" | "Manual Review" | "Decline"

  if (score <= 0.10) {
    // 0% to 10%: Ultra-safe pool (default_rate_at_0_1 = 1.47%)
    risk = "Low";
    color = "green";
    decision = "Approve";
  } else if (score <= 0.20) {
    // 10% to 20%: Moderate risk pool. Great for higher interest rates or manual underwriting review
    risk = "Medium";
    color = "orange";
    decision = "Manual Review";
  } else {
    // 20%+: Toxic zone (Enforces 0.2 MLflow cutoff threshold to protect portfolio)
    risk = "High";
    color = "red";
    decision = "Decline";
  }

  const topFeatures = result.shap_features//.slice(0, 8);
  const maxAbsShap = Math.max(
    ...topFeatures.map((f) => Math.abs(f.shap_value))
  );

  return (
    <div style={{ marginTop: "2rem" }}>
      <h2>Risk Assessment</h2>

      <p>
        <strong>Predicted Default Probability:</strong> {percentage}%
      </p>
      <p>
        <strong>Risk level:</strong>{" "}
        <span style={{ color }}>{risk}</span>
      </p>
      <p>
        <strong>Decision:</strong>{" "}
        <span style={{ color }}>{decision}</span>
      </p>
      {/* <p style={{ fontSize: "0.8rem", color: "gray" }}>
        Base value: {(result.base_value).toFixed(1)}
      </p> */}

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