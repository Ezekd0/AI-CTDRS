import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api",
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
