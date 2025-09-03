/**
 * Enhanced error handling utilities for NewsNeuron
 * Provides comprehensive error management and user feedback
 */

import { useAppStore } from "@/stores/app";

export class ErrorHandler {
  static instance = null;

  constructor() {
    if (ErrorHandler.instance) {
      return ErrorHandler.instance;
    }
    ErrorHandler.instance = this;
    this.appStore = null;
  }

  init(appStore) {
    this.appStore = appStore;
  }

  /**
   * Handle API errors with user-friendly messages
   */
  handleApiError(error, context = "") {
    if (!this.appStore) {
      console.error("ErrorHandler not initialized with app store");
      return;
    }

    let message = "An unexpected error occurred";
    let title = "Error";

    if (error?.response) {
      // HTTP error responses
      const { status, data } = error.response;

      switch (status) {
        case 400:
          title = "Invalid Request";
          message =
            data?.message ||
            "The request was invalid. Please check your input.";
          break;
        case 401:
          title = "Authentication Required";
          message = "Please log in to continue.";
          break;
        case 403:
          title = "Access Denied";
          message = "You do not have permission to perform this action.";
          break;
        case 404:
          title = "Not Found";
          message = data?.message || "The requested resource was not found.";
          break;
        case 422:
          title = "Validation Error";
          message =
            this.formatValidationErrors(data?.detail) ||
            "Please check your input.";
          break;
        case 429:
          title = "Too Many Requests";
          message = "Please wait a moment before trying again.";
          break;
        case 500:
          title = "Server Error";
          message = "A server error occurred. Please try again later.";
          break;
        case 503:
          title = "Service Unavailable";
          message =
            "The service is temporarily unavailable. Please try again later.";
          break;
        default:
          title = "Network Error";
          message = data?.message || `Request failed with status ${status}`;
      }
    } else if (error?.request) {
      // Network error
      title = "Connection Error";
      message =
        "Unable to connect to the server. Please check your internet connection.";
    } else if (error?.message) {
      // JavaScript error
      title = "Application Error";
      message = error.message;
    }

    // Add context if provided
    if (context) {
      message = `${context}: ${message}`;
    }

    // Log error for debugging
    console.error("Error handled:", {
      error,
      context,
      message,
      stack: error?.stack,
    });

    // Show user notification
    this.appStore.showError(message, title);

    return { title, message };
  }

  /**
   * Format validation errors from FastAPI
   */
  formatValidationErrors(details) {
    if (!Array.isArray(details)) {
      return details;
    }

    return details
      .map((error) => {
        const field = error.loc?.slice(-1)[0] || "field";
        return `${field}: ${error.msg}`;
      })
      .join(", ");
  }

  /**
   * Handle async operations with error handling
   */
  async withErrorHandling(operation, context = "", options = {}) {
    const {
      showLoading = false,
      suppressErrors = false,
      fallbackValue = null,
    } = options;

    try {
      if (showLoading && this.appStore) {
        this.appStore.setLoading(true);
      }

      const result = await operation();
      return { success: true, data: result, error: null };
    } catch (error) {
      if (!suppressErrors) {
        this.handleApiError(error, context);
      }

      return {
        success: false,
        data: fallbackValue,
        error: this.formatError(error),
      };
    } finally {
      if (showLoading && this.appStore) {
        this.appStore.setLoading(false);
      }
    }
  }

  /**
   * Format error for programmatic use
   */
  formatError(error) {
    return {
      message: error?.message || "Unknown error",
      status: error?.response?.status || null,
      code: error?.code || null,
      data: error?.response?.data || null,
    };
  }

  /**
   * Retry failed operations with exponential backoff
   */
  async retryOperation(operation, maxRetries = 3, baseDelay = 1000) {
    let lastError = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;

        if (attempt === maxRetries) {
          throw error;
        }

        // Don't retry on certain errors
        if (
          error?.response?.status &&
          [400, 401, 403, 404, 422].includes(error.response.status)
        ) {
          throw error;
        }

        // Exponential backoff
        const delay = baseDelay * Math.pow(2, attempt - 1);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }

  /**
   * Handle validation errors for forms
   */
  handleValidationError(error, formRef = null) {
    if (error?.response?.status === 422 && error?.response?.data?.detail) {
      const errors = error.response.data.detail;

      if (Array.isArray(errors) && formRef) {
        // Set field-specific errors if form reference is available
        errors.forEach((err) => {
          const field = err.loc?.slice(-1)[0];
          if (field && formRef.setFieldError) {
            formRef.setFieldError(field, err.msg);
          }
        });
      }

      return this.formatValidationErrors(errors);
    }

    return this.handleApiError(error, "Validation failed");
  }
}

// Create singleton instance
export const errorHandler = new ErrorHandler();

/**
 * Composable for error handling in Vue components
 */
export function useErrorHandler() {
  const appStore = useAppStore();

  // Initialize if not already done
  if (!errorHandler.appStore) {
    errorHandler.init(appStore);
  }

  return {
    handleError: (error, context) =>
      errorHandler.handleApiError(error, context),
    withErrorHandling: (operation, context, options) =>
      errorHandler.withErrorHandling(operation, context, options),
    retryOperation: (operation, maxRetries, baseDelay) =>
      errorHandler.retryOperation(operation, maxRetries, baseDelay),
    handleValidationError: (error, formRef) =>
      errorHandler.handleValidationError(error, formRef),
  };
}

/**
 * Global error handler for unhandled errors
 */
export function setupGlobalErrorHandling() {
  // Handle unhandled promise rejections
  window.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled promise rejection:", event.reason);
    errorHandler.handleApiError(event.reason, "Unhandled promise rejection");
    event.preventDefault();
  });

  // Handle global JavaScript errors
  window.addEventListener("error", (event) => {
    console.error("Global error:", event.error);
    if (event.error) {
      errorHandler.handleApiError(event.error, "Application error");
    }
  });
}

export default errorHandler;
