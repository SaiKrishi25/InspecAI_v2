import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Analytics API
export const getAnalyticsOverview = async (days: number = 30) => {
  const response = await api.get(`/api/analytics/overview?days=${days}`);
  return response.data;
};

export const getTimeSeriesData = async (days: number = 30) => {
  const response = await api.get(`/api/analytics/time-series?days=${days}`);
  return response.data;
};

export const getDefectDistribution = async () => {
  const response = await api.get('/api/analytics/defect-distribution');
  return response.data;
};

// Detections API
export const getRecentDetections = async (limit: number = 10) => {
  const response = await api.get(`/api/detections?limit=${limit}`);
  return response.data;
};

export const getDetectionById = async (detectionId: string) => {
  const response = await api.get(`/api/detections/${detectionId}`);
  return response.data;
};

// Reports API
export const getAllReports = async (limit: number = 100, offset: number = 0) => {
  const response = await api.get(`/api/reports/list?limit=${limit}&offset=${offset}`);
  return response.data;
};

export const searchReports = async (searchTerm: string = '', startDate: string = '', endDate: string = '') => {
  const response = await api.get('/api/reports/search', {
    params: { search: searchTerm, start_date: startDate, end_date: endDate },
  });
  return response.data;
};

export const getReportById = async (reportId: string) => {
  const response = await api.get(`/api/reports/${reportId}`);
  return response.data;
};

// Camera API
export const getCameraStatus = async () => {
  const response = await api.get('/api/camera/status');
  return response.data;
};

export const captureAndInfer = async (detectionMode: string = 'fasterrcnn_only') => {
  const response = await api.post('/api/capture', {
    detection_mode: detectionMode,
  });
  return response.data;
};

export default api;

