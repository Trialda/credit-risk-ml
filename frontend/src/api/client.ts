import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface PredictRequest {
  // Financial Amounts
  amt_credit: number;
  amt_income_total: number;
  amt_annuity: number;
  amt_goods_price: number;

  // Timeline & Demographics (Days)
  days_birth: number;
  days_employed: number;
  days_registration: number;
  days_id_publish: number;

  // Family Metrics
  cnt_children: number;
  cnt_fam_members: number;

  // Engine / Engineered Ratios
  credit_income_ratio: number;
  annuity_income_ratio: number;
  credit_term: number;
  age_years: number;
  employment_years: number;
  employment_to_age_ratio: number;

  // Categorical Strings
  name_contract_type: string;
  code_gender: string;
  name_income_type: string;
  name_education_type: string;
  name_family_status: string;
  name_housing_type: string;
  occupation_type: string;
  organization_type: string;

  // Bureau History Metrics
  bureau_count: number;
  bureau_mean_days_credit: number;
  bureau_total_credit: number;
  bureau_total_debt: number;
  bureau_mean_overdue: number;
  bureau_max_overdue_days: number;
  bureau_active_count: number;
  bureau_closed_count: number;
  bureau_bal_count: number;
  bureau_bal_closed_count: number;

  // Previous Application Metrics
  prev_app_count: number;
  prev_app_approved_count: number;
  prev_app_refused_count: number;
  prev_app_mean_credit: number;
  prev_app_mean_term: number;

  // Installment Behavior Metrics
  installments_count: number;
  installments_mean_payment_diff: number;
  installments_max_payment_diff: number;
  installments_mean_days_late: number;
  installments_max_days_late: number;
  installments_late_count: number;

  // Credit Card History Metrics
  credit_card_count: number;
  credit_card_mean_balance: number;
  credit_card_mean_utilisation: number;
  credit_card_max_utilisation: number;

  // POS / Cash Metrics
  pos_cash_count: number;
  pos_cash_mean_dpd: number;
  pos_cash_max_dpd: number;
  pos_cash_late_count: number;
}

export interface PredictResponse {
  request_id: string;
  risk_score: number;
}

export interface ShapFeature {
  feature: string;
  value: number | string;
  shap_value: number;
}

export interface ExplainResponse {
  request_id: string;
  risk_score: number;
  base_value: number;
  shap_features: ShapFeature[];
}

export async function predict(
  payload: PredictRequest
): Promise<PredictResponse> {
  /**
   * Send a prediction request to the backend.
   * Throws an AxiosError if the request fails.
   */
  const response = await apiClient.post<PredictResponse>("/predict", payload);
  return response.data;
}

export async function explainPrediction(
  features: PredictRequest
): Promise<ExplainResponse> {
  /**
   * Send an explanation request for a prior prediction.
   * Throws an AxiosError if the request fails.
   */
  const response = await apiClient.post<ExplainResponse>("/explain", {
    features,
  });
  return response.data;
}

export async function enrichCustomer(
  customerId: number
): Promise<PredictRequest> {
  const response = await apiClient.get<PredictRequest>(
    `/enrich/${customerId}`
  );
  return response.data;
}

export interface SimulateRequest {
  n_requests: number;
  duration_seconds: number;
  normal_fraction: number;
  drift_feature: string;
  drift_magnitude: number;
  drift_speed: "sudden" | "gradual";
}

export interface SimulateStatus {
  running: boolean;
  completed: number;
  total: number;
  failed: number;
  cancelled: boolean;
  progress: number;
  params: SimulateRequest;
}

export async function startSimulation(
  params: SimulateRequest
): Promise<void> {
  await apiClient.post("/simulate", params);
}

export async function stopSimulation(): Promise<void> {
  await apiClient.delete("/simulate");
}

export async function getSimulationStatus(): Promise<SimulateStatus> {
  const response = await apiClient.get<SimulateStatus>("/simulate/status");
  return response.data;
}

export async function triggerDriftComputation(): Promise<void> {
  await apiClient.post("/drift");
}