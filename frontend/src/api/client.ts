import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface PredictRequest {
  amt_credit: number;
  amt_income_total: number;
  days_birth: number;
  days_employed: number;
  cnt_children: number;
  amt_annuity: number;
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
  request_id: string,
  features: PredictRequest
): Promise<ExplainResponse> {
  /**
   * Send an explanation request for a prior prediction.
   * Throws an AxiosError if the request fails.
   */
  const response = await apiClient.post<ExplainResponse>("/explain", {
    request_id,
    features,
  });
  return response.data;
}