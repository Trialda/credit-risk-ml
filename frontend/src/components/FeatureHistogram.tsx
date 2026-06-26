import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { getDriftFeatureList, getFeatureHistogram, type HistogramResult } from "../api/client";

interface ChartBin {
  bin: string;
  training: number;
  production: number;
}

function buildChartData(result: HistogramResult): ChartBin[] {
  const { bin_edges, training_proportions, production_proportions } = result;
  return training_proportions.map((trainProp, i) => {
    const low = bin_edges[i];
    const high = bin_edges[i + 1];
    return {
      bin: `${low.toFixed(0)}–${high.toFixed(0)}`,
      training: Math.round(trainProp * 1000) / 10,
      production: Math.round((production_proportions[i] ?? 0) * 1000) / 10,
    };
  });
}

interface Props {
  selected: string;
  onSelect: (feature: string) => void;
}

export default function FeatureHistogram({ selected, onSelect }: Props) {
  const [features, setFeatures] = useState<string[]>([]);
  const [chartData, setChartData] = useState<ChartBin[]>([]);
  const [nSamples, setNSamples] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDriftFeatureList()
      .then(setFeatures)
      .catch(() => setError("Could not load feature list"));
  }, []);

  useEffect(() => {
    if (!selected) return;

    let cancelled = false;

    async function fetchHistogram() {
      setLoading(true);
      setError(null);

      try {
        const result = await getFeatureHistogram(selected);
        if (cancelled) return;
        setChartData(buildChartData(result));
        setNSamples(result.n_samples);
      } catch {
        if (!cancelled) setError("No production data yet — run a simulation first");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchHistogram();

    return () => {
      cancelled = true;
    };
  }, [selected]);

  return (
    <div>
      <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>
        Feature distribution comparison
      </h3>
      <p style={{ fontSize: "0.8rem", color: "gray" }}>
        Training distribution (blue) vs recent production traffic (orange),
        binned identically. Click any bar above to inspect that feature,
        or choose one directly below.
      </p>

      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        style={{ padding: "0.4rem", marginBottom: "1rem", width: "100%" }}
      >
        {features.map((f) => (
          <option key={f} value={f}>{f}</option>
        ))}
      </select>

      {error && <p style={{ color: "#d62728", fontSize: "0.85rem" }}>{error}</p>}

      <div style={{ position: "relative", minHeight: 320 }}>
        {loading && (
          <div style={{
            position: "absolute", inset: 0, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.03)", zIndex: 1, fontSize: "0.85rem",
          }}>
            Loading...
          </div>
        )}

        {!error && chartData.length > 0 && (
          <>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={60} />
                <YAxis label={{ value: "% of samples", angle: -90, position: "insideLeft", fontSize: 11 }} />
                <Tooltip 
                  formatter={(v: number | string | undefined | readonly (string | number)[]) => {
                    const safeVal = Array.isArray(v) ? v[0] : v;
                    return `${safeVal ?? 0}%`;
                  }} 
                />
                <Legend />
                <Bar dataKey="training" fill="#1f77b4" fillOpacity={0.55} name="Training" />
                <Bar dataKey="production" fill="#ff7f0e" fillOpacity={0.55} name="Production (recent)" />
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: "0.75rem", color: "gray" }}>
              Based on {nSamples} recent production samples, binned against
              training's stored histogram edges.
            </p>
          </>
        )}
      </div>
    </div>
  );
}