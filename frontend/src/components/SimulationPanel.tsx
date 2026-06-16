import { useState, useEffect, useRef } from "react";
import {
  startSimulation,
  stopSimulation,
  getSimulationStatus,
  triggerDriftComputation,
  type SimulateRequest,
} from "../api/client";

const DRIFT_FEATURES = [
  "amt_credit",
  "amt_income_total",
  "amt_annuity",
  "days_birth",
  "days_employed",
];

const defaultParams: SimulateRequest = {
  n_requests: 200,
  duration_seconds: 120,
  normal_fraction: 0.5,
  drift_feature: "amt_credit",
  drift_magnitude: 3.0,
  drift_speed: "gradual",
};

export default function SimulationPanel() {
  const [params, setParams] = useState<SimulateRequest>(defaultParams);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [completed, setCompleted] = useState(0);
  const [total, setTotal] = useState(0);
  const [failed, setFailed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [driftTriggered, setDriftTriggered] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  function startPolling() {
    pollRef.current = setInterval(async () => {
      try {
        const status = await getSimulationStatus();
        setProgress(status.progress);
        setCompleted(status.completed);
        setTotal(status.total);
        setFailed(status.failed);

        if (!status.running) {
          setRunning(false);
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    }, 2000);
  }

  async function handleStart() {
    setError(null);
    setDriftTriggered(false);
    try {
      await startSimulation(params);
      setRunning(true);
      setProgress(0);
      setCompleted(0);
      setFailed(0);
      startPolling();
    } catch {
      setError("Failed to start simulation. Is the backend running?");
    }
  }

  async function handleStop() {
    try {
      await stopSimulation();
      setRunning(false);
      if (pollRef.current) clearInterval(pollRef.current);
    } catch {
      setError("Failed to stop simulation.");
    }
  }

  async function handleTriggerDrift() {
    try {
      await triggerDriftComputation();
      setDriftTriggered(true);
    } catch {
      setError("Drift computation failed — collect more data first.");
    }
  }

  function handleChange(
    key: keyof SimulateRequest,
    value: string | number
  ) {
    setParams({ ...params, [key]: value });
  }

  const progressPct = Math.round(progress * 100);

  return (
    <div>
      <h2>Traffic Simulation</h2>
      <p style={{ fontSize: "0.85rem", color: "gray" }}>
        Generate synthetic loan applications to populate Grafana dashboards
        and test drift detection. Mix normal and drifted traffic to observe
        PSI gauges respond in real time.
      </p>

      <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
        <label style={{ fontSize: "0.85rem" }}>
          Total requests: {params.n_requests}
          <input type="range" min={1} max={2000} step={1}
            value={params.n_requests}
            onChange={(e) => handleChange("n_requests", parseInt(e.target.value))}
            style={{ width: "100%", display: "block" }} />
        </label>

        <label style={{ fontSize: "0.85rem" }}>
          Duration: {params.duration_seconds}s
          <input type="range" min={1} max={3600} step={1}
            value={params.duration_seconds}
            onChange={(e) => handleChange("duration_seconds", parseInt(e.target.value))}
            style={{ width: "100%", display: "block" }} />
        </label>

        <label style={{ fontSize: "0.85rem" }}>
          Normal traffic fraction: {Math.round(params.normal_fraction * 100)}%
          <input type="range" min={0} max={1} step={0.05}
            value={params.normal_fraction}
            onChange={(e) => handleChange("normal_fraction", parseFloat(e.target.value))}
            style={{ width: "100%", display: "block" }} />
        </label>

        <label style={{ fontSize: "0.85rem" }}>
          Drift magnitude: {params.drift_magnitude}σ
          <input type="range" min={0.1} max={10} step={0.1}
            value={params.drift_magnitude}
            onChange={(e) => handleChange("drift_magnitude", parseFloat(e.target.value))}
            style={{ width: "100%", display: "block" }} />
        </label>

        <label style={{ fontSize: "0.85rem" }}>
          Feature to drift:
          <select
            value={params.drift_feature}
            onChange={(e) => handleChange("drift_feature", e.target.value)}
            style={{ display: "block", width: "100%", padding: "0.3rem",
              marginTop: "0.2rem" }}
          >
            {DRIFT_FEATURES.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </label>

        <label style={{ fontSize: "0.85rem" }}>
          Drift speed:
          <select
            value={params.drift_speed}
            onChange={(e) =>
              handleChange("drift_speed", e.target.value as "sudden" | "gradual")
            }
            style={{ display: "block", width: "100%", padding: "0.3rem",
              marginTop: "0.2rem" }}
          >
            <option value="gradual">Gradual (increases over time)</option>
            <option value="sudden">Sudden (immediate full drift)</option>
          </select>
        </label>
      </div>

      {running && (
        <div style={{ marginBottom: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between",
            fontSize: "0.85rem", marginBottom: "4px" }}>
            <span>Progress: {completed}/{total} requests</span>
            <span>{progressPct}%</span>
          </div>
          <div style={{ height: "8px", background: "#eee", borderRadius: "4px" }}>
            <div style={{
              height: "100%",
              width: `${progressPct}%`,
              background: "#1f77b4",
              borderRadius: "4px",
              transition: "width 0.5s ease",
            }} />
          </div>
          {failed > 0 && (
            <p style={{ fontSize: "0.75rem", color: "orange", margin: "4px 0 0" }}>
              {failed} requests failed
            </p>
          )}
        </div>
      )}

      {error && <p style={{ color: "red", fontSize: "0.85rem" }}>{error}</p>}

      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        {!running ? (
          <button onClick={handleStart}>Start Simulation</button>
        ) : (
          <button onClick={handleStop}
            style={{ background: "#d62728", color: "white", border: "none",
              padding: "0.4rem 0.8rem", borderRadius: "4px", cursor: "pointer" }}>
            Stop
          </button>
        )}

        <button
          onClick={handleTriggerDrift}
          disabled={running}
          style={{ opacity: running ? 0.5 : 1 }}
        >
          Compute Drift Now
        </button>
      </div>

      {driftTriggered && (
        <p style={{ fontSize: "0.75rem", color: "green", marginTop: "0.5rem" }}>
          Drift computation triggered — check Grafana for updated PSI gauges.
        </p>
      )}

      <p style={{ fontSize: "0.75rem", color: "gray", marginTop: "1rem" }}>
        View results at{" "}
        <a href="http://localhost:3000" target="_blank" rel="noreferrer">
          Grafana dashboard
        </a>
      </p>
    </div>
  );
}