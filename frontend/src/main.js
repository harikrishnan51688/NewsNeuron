import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import router from "./router";

// Styles
import "./style.css";

// Error handling
import { setupGlobalErrorHandling, errorHandler } from "./utils/errorHandler";

// Create Vue app
const app = createApp(App);

// Create pinia instance
const pinia = createPinia();

// Use plugins
app.use(pinia);
app.use(router);

// Setup global error handling
setupGlobalErrorHandling();

// Enhanced global error handler
app.config.errorHandler = (err, instance, info) => {
  console.error("Vue error:", { err, info, instance });

  // Use our error handler
  if (errorHandler.appStore) {
    errorHandler.handleApiError(err, `Vue error: ${info}`);
  } else {
    // Fallback for early errors before store is ready
    console.error("Error occurred before error handler was initialized:", err);
  }

  // Send to external error tracking service (e.g., Sentry)
  if (import.meta.env.VITE_SENTRY_DSN) {
    // Sentry.captureException(err, { extra: { info, instance } })
  }
};

// Global properties
app.config.globalProperties.$appName =
  import.meta.env.VITE_APP_NAME || "NewsNeuron";
app.config.globalProperties.$appVersion =
  import.meta.env.VITE_APP_VERSION || "1.0.0";

// Performance monitoring
if (import.meta.env.VITE_ENABLE_DEBUG === "true") {
  app.config.performance = true;
}

// Mount the app
app.mount("#app");
