import { useState } from "react";
import { predict, type PredictRequest, type PredictResponse } from "../api/client";

interface Props {
  onResult: (result: PredictResponse) => void;
}

const defaultForm: PredictRequest = {
  amt_credit: 500000,
  amt_income_total: 150000,
  days_birth: -12000,
  days_employed: -2000,
  cnt_children: 0,
  amt_annuity: 24700,
};

export default function LoanForm({ onResult }: Props) {
  const [form, setForm] = useState<PredictRequest>(defaultForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setForm({ ...form, [e.target.name]: Number(e.target.value) });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await predict(form);
      onResult(result);
    } catch (err) {
      console.error("API Error:", err);
      setError("Prediction failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <h2>Loan Application</h2>

      {Object.entries(form).map(([key, value]) => (
        <div key={key}>
          <label htmlFor={key}>{key}</label>
          <input
            id={key}
            name={key}
            type="number"
            value={value}
            onChange={handleChange}
          />
        </div>
      ))}

      {error && <p style={{ color: "red" }}>{error}</p>}

      <button type="submit" disabled={loading}>
        {loading ? "Predicting..." : "Get Risk Score"}
      </button>
    </form>
  );
}