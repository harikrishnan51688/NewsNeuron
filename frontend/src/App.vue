<template>
  <div id="app" class="neural-app bg-neuron-bg-primary text-neuron-text-primary flex flex-col min-h-screen" lang="en">
    <!-- Screen Reader Support -->
    <ScreenReaderSupport />
    
    <!-- Neural Navigation -->
    <AppNavigation id="navigation" role="navigation" aria-label="Main navigation" />

    <!-- Main Content Area -->
    <main id="main-content" class="main-content flex-1" role="main" tabindex="-1">
      <div class="content-wrapper">
        <transition name="neural-fade" mode="out-in">
          <RouterView />
        </transition>
      </div>
    </main>

    <!-- Global Loading Indicator -->
    <LoadingIndicator v-if="isLoading" aria-label="Loading content" />

    <!-- Toast Notifications -->
    <ToastContainer role="region" aria-label="Notifications" aria-live="polite" />
  </div>
</template>

<script setup>
import { computed } from "vue";
import { RouterView } from "vue-router";
import { useAppStore } from "@/stores/app";

// Components
import AppNavigation from "@/components/layout/AppNavigation.vue";
import LoadingIndicator from "@/components/ui/LoadingIndicator.vue";
import ToastContainer from "@/components/ui/ToastContainer.vue";
import ScreenReaderSupport from "@/components/ui/ScreenReaderSupport.vue";

// Store
const appStore = useAppStore();

// Computed
const isLoading = computed(() => appStore.isLoading);

// Initialize app
appStore.initialize();
</script>

<style lang="postcss">
/* Global neural styles */
.neural-app {
  font-family: 'Manrope', 'Inter', system-ui, sans-serif;
}

/* Main content area */
.main-content {
  @apply transition-all duration-300;
}

.content-wrapper {
  @apply min-h-full;
}

/* Neural fade transition */
.neural-fade-enter-active,
.neural-fade-leave-active {
  @apply transition-all duration-500;
}

.neural-fade-enter-from {
  @apply opacity-0 transform translate-x-4;
}

.neural-fade-leave-to {
  @apply opacity-0 transform -translate-x-4;
}

/* Scrollbar styling for neural theme */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: rgba(42, 45, 53, 0.3);
}

::-webkit-scrollbar-thumb {
  background: rgba(0, 212, 255, 0.3);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 212, 255, 0.5);
}

/* Firefox scrollbar */
html {
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 212, 255, 0.3) rgba(42, 45, 53, 0.3);
}

/* Selection styling */
::selection {
  background: rgba(0, 212, 255, 0.3);
  color: #F5F7FA;
}

/* Focus ring improvements */
:focus-visible {
  @apply outline-none;
  box-shadow: 0 0 0 2px rgba(0, 167, 225, 0.6);
}

/* Enhanced focus for interactive elements */
button:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
[tabindex]:focus-visible {
  @apply outline-none;
  box-shadow: 0 0 0 2px rgba(0, 167, 225, 0.6);
}

/* Error state focus */
.error:focus-visible,
[aria-invalid="true"]:focus-visible {
  @apply outline-none;
  box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.6);
}

/* Skip links focus */
.skip-link:focus-visible {
  @apply outline-none;
  box-shadow: 0 0 0 3px rgba(0, 167, 225, 0.8);
}

/* Smooth scrolling */
html {
  scroll-behavior: smooth;
}

/* Remove default margins and paddings */
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  @apply bg-neuron-bg-primary text-neuron-text-primary;
}

/* Ensure proper text rendering */
body {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

/* Hide scrollbar on mobile for cleaner look */
@media (max-width: 768px) {
  .main-content {
    @apply ml-0;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .neural-fade-enter-active,
  .neural-fade-leave-active,
  .main-content {
    transition: none !important;
  }
  
  html {
    scroll-behavior: auto;
  }
  
  /* Disable all animations */
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* High contrast support */
@media (prefers-contrast: high) {
  .neural-app {
    filter: contrast(150%);
  }
  
  :focus-visible {
    outline: 3px solid !important;
    outline-offset: 2px !important;
  }
}

/* Forced colors mode support */
@media (forced-colors: active) {
  .neural-app {
    forced-color-adjust: none;
  }
  
  :focus-visible {
    outline: 2px solid ButtonText;
    outline-offset: 2px;
  }
}

/* Color scheme support */
@media (prefers-color-scheme: dark) {
  /* Already using dark theme by default */
}

/* Touch target sizes for mobile accessibility */
@media (max-width: 768px) {
  button,
  [role="button"],
  a,
  input,
  textarea,
  select {
    min-height: 44px;
    min-width: 44px;
  }
}

/* Screen reader utilities */
.sr-only {
  @apply absolute w-px h-px p-0 -m-px overflow-hidden;
  @apply whitespace-nowrap border-0;
  clip: rect(0, 0, 0, 0);
}

.sr-only-focusable:focus {
  @apply static w-auto h-auto p-1 m-0 overflow-visible;
  @apply whitespace-normal;
  clip: auto;
}
</style>
