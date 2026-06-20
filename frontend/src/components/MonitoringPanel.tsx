import { useState, useMemo } from "react";
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { getDriftResults, type DriftResult } from "../api/client";
import FeatureHistogram from "./FeatureHistogram";

type SortMode = "psi_desc" | "alphabetical";

const PSI_STABLE = 0.1;
const PSI_WARNING = 0.2;

function psiColor(psi: number): string {
  if (psi >= PSI_WARNING) return "#d62728";
  if (psi >= PSI_STABLE) return "#EAB839";
  return "#2ca02c";
}

export default function MonitoringPanel() {
  const [result, setResult] = useState<DriftResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortMode, setSortMode] = useState<SortMode>("psi_desc");
  const [showCount, setShowCount] = useState(15);
  const [selectedFeature, setSelectedFeature] = useState<string>("amt_credit");
  async function handleComputeDrift() {
    setLoading(true);
    setError(null);
    try {
      const data = await getDriftResults();
      setResult(data);
    } catch {
      setError(
        "Drift computation unavailable. Ensure training reference exists " +
        "and enough inference data has been collected (run a simulation first)."
      );
    } finally {
      setLoading(false);
    }
  }

  const chartData = useMemo(() => {
    if (!result) return [];

    const entries = Object.entries(result.feature_psi).map(([feature, psi]) => ({
      feature,
      psi: Math.round(psi * 1000) / 1000,
    }));

    if (sortMode === "psi_desc") {
      entries.sort((a, b) => b.psi - a.psi);
    } else {
      entries.sort((a, b) => a.feature.localeCompare(b.feature));
    }

    return entries.slice(0, showCount);
  }, [result, sortMode, showCount]);

  const driftedCount = useMemo(() => {
    if (!result) return 0;
    return Object.values(result.feature_psi).filter((p) => p >= PSI_WARNING).length;
  }, [result]);

  const statusColor = driftedCount === 0 ? "#2ca02c" : driftedCount < 5 ? "#EAB839" : "#d62728";
  const statusText = driftedCount === 0
    ? "No significant drift detected"
    : `${driftedCount} feature(s) showing significant drift`;

  return (
    <div>
      
      <h2>Drift Monitoring</h2>
      <p style={{ fontSize: "0.85rem", color: "gray" }}>
        Per-feature Population Stability Index against the training distribution.
        Run a simulation first to generate inference data, then compute drift.
      </p>

      <button onClick={handleComputeDrift} disabled={loading} style={{ marginBottom: "1rem" }}>
        {loading ? "Computing..." : "Compute Drift"}
      </button>

      {error && (
        <p style={{ color: "#d62728", fontSize: "0.85rem", marginBottom: "1rem" }}>
          {error}
        </p>
      )}

      {result && (
        <>
          <div style={{
            padding: "0.75rem 1rem",
            borderRadius: "6px",
            background: statusColor + "22",
            border: `1px solid ${statusColor}`,
            marginBottom: "1rem",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "0.5rem",
          }}>
            <span style={{ fontWeight: "bold", color: statusColor }}>
              {statusText}
            </span>
            <span style={{ fontSize: "0.8rem", color: "gray" }}>
              Based on {result.n_samples} recent inference requests
            </span>
          </div>

          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            <div style={{ padding: "0.75rem", background: "#f5f5f522", borderRadius: "6px", flex: 1 }}>
              <div style={{ fontSize: "0.75rem", color: "gray" }}>Score distribution PSI</div>
              <div style={{
                fontSize: "1.5rem",
                fontWeight: "bold",
                color: psiColor(result.score_drift.score_psi),
              }}>
                {result.score_drift.score_psi.toFixed(3)}
              </div>
            </div>
            <div style={{ padding: "0.75rem", background: "#f5f5f522", borderRadius: "6px", flex: 1 }}>
              <div style={{ fontSize: "0.75rem", color: "gray" }}>Mean predicted score</div>
              <div style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
                {(result.score_drift.score_mean * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ padding: "0.75rem", background: "#f5f5f522", borderRadius: "6px", flex: 1 }}>
              <div style={{ fontSize: "0.75rem", color: "gray" }}>Sample size</div>
              <div style={{ fontSize: "1.5rem", fontWeight: "bold" }}>
                {result.n_samples}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between",
            alignItems: "center", marginBottom: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}>
            <h3 style={{ margin: 0, fontSize: "1rem" }}>Per-feature PSI</h3>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <select value={sortMode} onChange={(e) => setSortMode(e.target.value as SortMode)}
                style={{ fontSize: "0.8rem", padding: "0.3rem" }}>
                <option value="psi_desc">Sort by PSI (highest first)</option>
                <option value="alphabetical">Sort alphabetically</option>
              </select>
              <select value={showCount} onChange={(e) => setShowCount(parseInt(e.target.value))}
                style={{ fontSize: "0.8rem", padding: "0.3rem" }}>
                <option value={10}>Show top 10</option>
                <option value={15}>Show top 15</option>
                <option value={25}>Show top 25</option>
                <option value={Object.keys(result.feature_psi).length}>Show all</option>
              </select>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 28)}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" />
              <YAxis
  type="category"
  dataKey="feature"
  width={180}
  tick={{ fontSize: 11 }}
  interval={0}
/>
              <Tooltip
                formatter={(value: number) => [value.toFixed(3), "PSI"]}
              />
              <ReferenceLine x={PSI_STABLE} stroke="#EAB839" strokeDasharray="4 4" />
              <ReferenceLine x={PSI_WARNING} stroke="#d62728" strokeDasharray="4 4" />
<Bar dataKey="psi" radius={[0, 4, 4, 0]}>
  {chartData.map((entry, index) => (
    <Cell
      key={`cell-${index}`}
      fill={psiColor(entry.psi)}
      onClick={() => setSelectedFeature(entry.feature)}
      style={{ cursor: "pointer" }}
    />
  ))}
</Bar>
            </BarChart>
          </ResponsiveContainer>

          <p style={{ fontSize: "0.75rem", color: "gray", marginTop: "0.5rem" }}>
            Yellow line: PSI {PSI_STABLE} (moderate shift). Red line: PSI {PSI_WARNING} (significant drift).
          </p>
        </>
      )}
      <FeatureHistogram selected={selectedFeature} onSelect={setSelectedFeature} />
    </div>
  );
}