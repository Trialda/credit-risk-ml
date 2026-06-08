import type { PredictResponse } from "../api/client";

interface Props {
  result: PredictResponse;
}

export default function RiskResult({ result }: Props) {
  const percentage = (result.risk_score * 100).toFixed(1);
  const risk = result.risk_score > 0.5 ? "High" : "Low";
  const color = result.risk_score > 0.5 ? "red" : "green";

  return (
    <div>
      <h2>Risk Assessment</h2>
      <p>
        <strong>Score:</strong> {percentage}%
      </p>
      <p>
        <strong>Risk level:</strong>{" "}
        <span style={{ color }}>{risk}</span>
      </p>
      <p style={{ fontSize: "0.8rem", color: "gray" }}>
        Request ID: {result.request_id}
      </p>
    </div>
  );
}