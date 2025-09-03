import { defineStore } from "pinia";
import { ref, computed } from "vue";

export const useAppStore = defineStore("app", () => {
  // State
  const isLoading = ref(false);
  const isDarkMode = ref(false);
  const sidebarOpen = ref(false);
  const notifications = ref([]);
  const appVersion = ref(import.meta.env.VITE_APP_VERSION || "1.0.0");
  const appName = ref(import.meta.env.VITE_APP_NAME || "NewsNeuron");

  // Error handling
  const error = ref(null);
  const errorHistory = ref([]);

  // App settings
  const settings = ref({
    theme: "light",
    language: "en",
    notifications: true,
    autoRefresh: true,
    refreshInterval: 30000, // 30 seconds
  });

  // Getters
  const hasError = computed(() => !!error.value);
  const notificationCount = computed(() => notifications.value.length);
  const unreadNotifications = computed(() =>
    notifications.value.filter((n) => !n.read),
  );

  // Actions
  function initialize() {
    // Load settings from localStorage
    loadSettings();

    // Set initial theme
    setTheme(settings.value.theme);

    // Check for saved preferences
    const savedSidebarState = localStorage.getItem("sidebarOpen");
    if (savedSidebarState !== null) {
      sidebarOpen.value = JSON.parse(savedSidebarState);
    }

    console.log(`${appName.value} v${appVersion.value} initialized`);
  }

  function setLoading(loading) {
    isLoading.value = loading;

    // Add a safety timeout to prevent infinite loading states
    if (loading) {
      // Clear loading after 30 seconds as a safety measure
      setTimeout(() => {
        if (isLoading.value) {
          console.warn("Loading state was stuck for 30 seconds, clearing automatically");
          isLoading.value = false;
        }
      }, 30000);
    }
  }

  function toggleSidebar() {
    sidebarOpen.value = !sidebarOpen.value;
    localStorage.setItem("sidebarOpen", JSON.stringify(sidebarOpen.value));
  }

  function setSidebarOpen(open) {
    sidebarOpen.value = open;
    localStorage.setItem("sidebarOpen", JSON.stringify(sidebarOpen.value));
  }

  function setTheme(theme) {
    settings.value.theme = theme;
    isDarkMode.value = theme === "dark";

    // Apply theme to document
    if (isDarkMode.value) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }

    saveSettings();
  }

  function toggleTheme() {
    const newTheme = isDarkMode.value ? "light" : "dark";
    setTheme(newTheme);
  }

  function setError(errorMessage, errorDetails = null) {
    const errorObj = {
      id: Date.now(),
      message: errorMessage,
      details: errorDetails,
      timestamp: new Date().toISOString(),
    };

    error.value = errorObj;
    errorHistory.value.unshift(errorObj);

    // Keep only last 10 errors
    if (errorHistory.value.length > 10) {
      errorHistory.value = errorHistory.value.slice(0, 10);
    }

    console.error("App Error:", errorMessage, errorDetails);
  }

  function clearError() {
    error.value = null;
  }

  function addNotification(notification) {
    const id = Date.now() + Math.random();
    const newNotification = {
      id,
      type: "info", // info, success, warning, error
      title: "",
      message: "",
      read: false,
      timestamp: new Date().toISOString(),
      autoHide: true,
      duration: 5000,
      ...notification,
    };

    notifications.value.unshift(newNotification);

    // Auto-hide if specified
    if (newNotification.autoHide) {
      setTimeout(() => {
        removeNotification(id);
      }, newNotification.duration);
    }

    return id;
  }

  function removeNotification(id) {
    const index = notifications.value.findIndex((n) => n.id === id);
    if (index !== -1) {
      notifications.value.splice(index, 1);
    }
  }

  function markNotificationRead(id) {
    const notification = notifications.value.find((n) => n.id === id);
    if (notification) {
      notification.read = true;
    }
  }

  function markAllNotificationsRead() {
    notifications.value.forEach((n) => (n.read = true));
  }

  function clearAllNotifications() {
    notifications.value = [];
  }

  function updateSetting(key, value) {
    if (key in settings.value) {
      settings.value[key] = value;
      saveSettings();

      // Handle special settings
      if (key === "theme") {
        setTheme(value);
      }
    }
  }

  function loadSettings() {
    try {
      const savedSettings = localStorage.getItem("newsNeuronSettings");
      if (savedSettings) {
        const parsed = JSON.parse(savedSettings);
        settings.value = { ...settings.value, ...parsed };
      }
    } catch (error) {
      console.warn("Failed to load settings:", error);
    }
  }

  function saveSettings() {
    try {
      localStorage.setItem(
        "newsNeuronSettings",
        JSON.stringify(settings.value),
      );
    } catch (error) {
      console.warn("Failed to save settings:", error);
    }
  }

  function resetSettings() {
    settings.value = {
      theme: "light",
      language: "en",
      notifications: true,
      autoRefresh: true,
      refreshInterval: 30000,
    };
    saveSettings();
    setTheme("light");
  }

  // Utility functions
  function showSuccess(message, title = "Success") {
    return addNotification({
      type: "success",
      title,
      message,
    });
  }

  function showError(message, title = "Error") {
    return addNotification({
      type: "error",
      title,
      message,
      autoHide: false, // Errors should be manually dismissed
    });
  }

  function showWarning(message, title = "Warning") {
    return addNotification({
      type: "warning",
      title,
      message,
    });
  }

  function showInfo(message, title = "Info") {
    return addNotification({
      type: "info",
      title,
      message,
    });
  }

  return {
    // State
    isLoading,
    isDarkMode,
    sidebarOpen,
    notifications,
    appVersion,
    appName,
    error,
    errorHistory,
    settings,

    // Getters
    hasError,
    notificationCount,
    unreadNotifications,

    // Actions
    initialize,
    setLoading,
    toggleSidebar,
    setSidebarOpen,
    setTheme,
    toggleTheme,
    setError,
    clearError,
    addNotification,
    removeNotification,
    markNotificationRead,
    markAllNotificationsRead,
    clearAllNotifications,
    updateSetting,
    loadSettings,
    saveSettings,
    resetSettings,
    showSuccess,
    showError,
    showWarning,
    showInfo,
  };
});
