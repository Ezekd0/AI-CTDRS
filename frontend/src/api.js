import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || "/api",
  timeout: 8000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});


api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("access_token");
      window.dispatchEvent(new Event("auth:logout"));
    }
    return Promise.reject(error);
  },
);

export async function login(email, password) {
  const response = await api.post("/auth/login", { email, password });
  localStorage.setItem("access_token", response.data.access_token);
  return response.data;
}

export function logout() {
  localStorage.removeItem("access_token");
  window.dispatchEvent(new Event("auth:logout"));
}

export function predictionErrorMessage(error) {
  if (error.response) {
    const { status, data } = error.response;
    const detail = typeof data?.detail === "string" ? data.detail : "Detection request failed";
    return `HTTP ${status}: ${detail}`;
  }
  if (error.code === "ECONNABORTED" || error.code === "ETIMEDOUT") {
    return "Detection request timed out. Check the backend before retrying; it may still be processing.";
  }
  if (error.code === "ERR_NETWORK" || error.message === "Network Error") {
    return "No readable response from the detection API. Check backend connectivity and CORS; the browser may have blocked an HTTP error response.";
  }
  return error.message || "Prediction failed.";
}
