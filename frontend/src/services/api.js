import axios from "axios";

// Create axios instance
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add API version to URL if not already present
    if (!config.url.includes("/api/")) {
      const version = import.meta.env.VITE_API_VERSION || "v1";
      config.url = `/api/${version}${config.url}`;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    // Import errorHandler dynamically to avoid circular imports
    // const { errorHandler } = await import("@/utils/errorHandler");

    // Let the error handler manage user notifications
    // The interceptor just logs and passes the error through
    console.warn("API Error intercepted:", {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      message: error.message,
    });

    // Don't show notifications here - let components handle it
    // This prevents duplicate error messages
    return Promise.reject(error);
  },
);

// API endpoints
export const chatAPI = {
  sendMessage: (data) => api.post("/chat/", data),
  getConversations: (params) => api.get("/chat/conversations", { params }),
  getConversationHistory: (conversationId, params) =>
    api.get(`/chat/conversations/${conversationId}/history`, { params }),
  deleteConversation: (conversationId) =>
    api.delete(`/chat/conversations/${conversationId}`),
};

export const flashcardsAPI = {
  generateFlashcards: (data) => api.post("/flashcards/", data),
  getFlashcards: (params) => api.get("/flashcards/", { params }),
  getFlashcardDetails: (flashcardId) => api.get(`/flashcards/${flashcardId}`),
  getTrendingTopics: (params) =>
    api.get("/flashcards/topics/trending", { params }),
};

export const searchAPI = {
  searchArticles: (data) => api.post("/search/", data),
  searchArticlesGet: (params) => api.get("/search/", { params }),
  getSearchSuggestions: (params) => api.get("/search/suggestions", { params }),
  searchEntities: (params) => api.get("/search/entities", { params }),
  findSimilarArticles: (articleId, params) =>
    api.get(`/search/similar/${articleId}`, { params }),
};

export const timelineAPI = {
  generateTimeline: (data) => api.post("/timeline/", data),
  getEntityTimeline: (entityName, params) =>
    api.get(`/timeline/${entityName}`, { params }),
  getTimelineSummary: (entityName, params) =>
    api.get(`/timeline/${entityName}/summary`, { params }),
  getRelatedEntitiesTimeline: (entityName, params) =>
    api.get(`/timeline/${entityName}/related`, { params }),
  getTrendingEvents: (params) =>
    api.get("/timeline/events/trending", { params }),
};

export const healthAPI = {
  checkHealth: () => api.get("/health"),
  getStatus: () => api.get("/"),
};

// Helper functions
export const handleAPIError = (error, defaultMessage = "An error occurred") => {
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  return defaultMessage;
};

export const downloadFile = async (url, filename) => {
  try {
    const response = await api.get(url, {
      responseType: "blob",
    });

    const blob = new Blob([response.data]);
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(downloadUrl);
  } catch (error) {
    console.error("Download failed:", error);
    throw error;
  }
};

export default api;
