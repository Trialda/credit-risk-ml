import { useState } from "react";
import type { ExplainResponse } from "./api/client";
import LoanForm from "./components/LoanForm";
import RiskResult from "./components/RiskResult";

export default function App() {
  const [result, setResult] = useState<ExplainResponse | null>(null);

  return (
    <main style={{ maxWidth: "600px", margin: "2rem auto", padding: "0 1rem" }}>
      <h1>Credit Risk Assessment</h1>
      <LoanForm onResult={setResult} />
      {result && <RiskResult result={result} />}
    </main>
  );
}