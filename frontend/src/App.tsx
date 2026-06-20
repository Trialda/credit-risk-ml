import { useState } from "react";
import type { ExplainResponse } from "./api/client";
import LoanForm from "./components/LoanForm";
import RiskResult from "./components/RiskResult";
import SimulationPanel from "./components/SimulationPanel";
import MonitoringPanel from "./components/MonitoringPanel";

type Tab = "predict" | "simulate" | "monitor";

export default function App() {
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("predict");

  const tabStyle = (tab: Tab) => ({
    padding: "0.5rem 1rem",
    border: "none",
    borderBottom: activeTab === tab ? "2px solid #1f77b4" : "2px solid transparent",
    background: "none",
    cursor: "pointer",
    fontWeight: activeTab === tab ? "bold" : "normal",
    color: activeTab === tab ? "#1f77b4" : "inherit",
  });

  return (
    <main style={{ maxWidth: "600px", margin: "2rem auto", padding: "0 1rem" }}>
      <h1>Credit Risk Assessment</h1>

      <div style={{ display: "flex", borderBottom: "1px solid #ddd",
        marginBottom: "1.5rem" }}>
        <button style={tabStyle("predict")} onClick={() => setActiveTab("predict")}>
          Predict
        </button>
        <button style={tabStyle("simulate")} onClick={() => setActiveTab("simulate")}>
          Simulate
        </button>
        <button style={tabStyle("monitor")} onClick={() => setActiveTab("monitor")}>
          Monitor
        </button>
      </div>

      {activeTab === "predict" && (
        <>
          <LoanForm onResult={setResult} />
          {result && <RiskResult result={result} />}
        </>
      )}

      {activeTab === "simulate" && <SimulationPanel />}
      {activeTab === "monitor" && <MonitoringPanel />}
    </main>
  );
}