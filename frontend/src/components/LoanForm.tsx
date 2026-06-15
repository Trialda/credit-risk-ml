import { useState } from "react";
import {
  enrichCustomer,
  explainPrediction,
  type ExplainResponse,
  type PredictRequest,
} from "../api/client";

interface Props {
  onResult: (result: ExplainResponse) => void;
}

const defaultPayload: PredictRequest = {
  amt_credit: 500000,
  amt_income_total: 150000,
  amt_annuity: 24700,
  amt_goods_price: 450000,
  days_birth: -12000,
  days_employed: -2000,
  days_registration: -5000,
  days_id_publish: -3000,
  cnt_children: 0,
  cnt_fam_members: 2,
  credit_income_ratio: 3.33,
  annuity_income_ratio: 0.16,
  credit_term: 20.24,
  age_years: 32.87,
  employment_years: 5.47,
  employment_to_age_ratio: 0.166,
  name_contract_type: "Cash loans",
  code_gender: "M",
  name_income_type: "Working",
  name_education_type: "Secondary / secondary special",
  name_family_status: "Married",
  name_housing_type: "House / apartment",
  occupation_type: "missing",
  organization_type: "Business Entity Type 3",
  bureau_count: 0, bureau_mean_days_credit: 0, bureau_total_credit: 0,
  bureau_total_debt: 0, bureau_mean_overdue: 0, bureau_max_overdue_days: 0,
  bureau_active_count: 0, bureau_closed_count: 0, bureau_bal_count: 0,
  bureau_bal_closed_count: 0, prev_app_count: 0, prev_app_approved_count: 0,
  prev_app_refused_count: 0, prev_app_mean_credit: 0, prev_app_mean_term: 0,
  installments_count: 0, installments_mean_payment_diff: 0,
  installments_max_payment_diff: 0, installments_mean_days_late: 0,
  installments_max_days_late: 0, installments_late_count: 0,
  credit_card_count: 0, credit_card_mean_balance: 0,
  credit_card_mean_utilisation: 0, credit_card_max_utilisation: 0,
  pos_cash_count: 0, pos_cash_mean_dpd: 0, pos_cash_max_dpd: 0,
  pos_cash_late_count: 0,
};

const MANUAL_FIELDS: (keyof PredictRequest)[] = [
  "amt_credit",
  "amt_income_total",
  "amt_annuity",
  "amt_goods_price",
  "days_birth",
  "days_employed",
  "cnt_children",
  "cnt_fam_members",
];

export default function LoanForm({ onResult }: Props) {
  const [form, setForm] = useState<PredictRequest>(defaultPayload);
  const [customerId, setCustomerId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [enriching, setEnriching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enriched, setEnriched] = useState(false);

  async function handleEnrich() {
    const id = parseInt(customerId);
    if (isNaN(id)) {
      setError("Enter a valid customer ID");
      return;
    }
    setEnriching(true);
    setError(null);
    try {
      const features = await enrichCustomer(id);
      setForm({ ...defaultPayload, ...features });
      setEnriched(true);
    } catch {
      setError(`Customer ${id} not found`);
    } finally {
      setEnriching(false);
    }
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = Number(e.target.value);
    const key = e.target.name as keyof PredictRequest;
    const updated = { ...form, [key]: value };

    updated.credit_income_ratio =
      updated.amt_income_total > 0
        ? updated.amt_credit / updated.amt_income_total
        : 0;
    updated.annuity_income_ratio =
      updated.amt_income_total > 0
        ? updated.amt_annuity / updated.amt_income_total
        : 0;
    updated.credit_term =
      updated.amt_annuity > 0
        ? updated.amt_credit / updated.amt_annuity
        : 0;
    updated.age_years = updated.days_birth / -365;
    updated.employment_years = updated.days_employed / -365;
    updated.employment_to_age_ratio =
      updated.days_birth !== 0
        ? updated.days_employed / updated.days_birth
        : 0;

    setForm(updated);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const explanation = await explainPrediction(form);
      onResult(explanation);
    } catch {
      setError("Request failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Loan Application</h2>

      <div style={{ marginBottom: "1rem", padding: "0.75rem",
        background: "var(--enrich-bg, #f5f5f5)", borderRadius: "6px" }}>
        <label style={{ display: "block", fontSize: "0.85rem",
          marginBottom: "0.4rem" }}>
          Customer ID (optional — pre-fills form from feature store)
        </label>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <input
            type="number"
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            placeholder="e.g. 100002"
            style={{ flex: 1, padding: "0.4rem" }}
          />
          <button
            type="button"
            onClick={handleEnrich}
            disabled={enriching}
          >
            {enriching ? "Fetching..." : "Fetch"}
          </button>
        </div>
        {enriched && (
          <p style={{ fontSize: "0.75rem", color: "green", margin: "0.3rem 0 0" }}>
            Form pre-populated from feature store. You can edit values before submitting.
          </p>
        )}
      </div>

      <form onSubmit={handleSubmit}>
        {MANUAL_FIELDS.map((key) => (
          <div key={key} style={{ marginBottom: "0.75rem" }}>
            <label htmlFor={key}
              style={{ display: "block", fontSize: "0.85rem" }}>
              {key}
            </label>
            <input
              id={key}
              name={key}
              type="number"
              value={form[key] as number}
              onChange={handleChange}
              style={{ width: "100%", padding: "0.4rem" }}
            />
          </div>
        ))}

        {error && <p style={{ color: "red" }}>{error}</p>}

        <button type="submit" disabled={loading}>
          {loading ? "Analysing..." : "Get Risk Score"}
        </button>
      </form>
    </div>
  );
}