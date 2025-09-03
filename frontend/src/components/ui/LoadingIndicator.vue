<template>
  <div 
    class="loading-indicator fixed inset-0 bg-neuron-bg-primary/80 backdrop-blur-sm z-50 flex items-center justify-center"
    role="status"
    aria-label="Loading"
  >
    <div class="loading-content bg-neuron-bg-content border border-neuron-border rounded-lg p-8 shadow-neuron-lg">
      <!-- Main loader -->
      <div class="flex flex-col items-center space-y-4">
        <SynapseLoader size="lg" :show-text="true" loading-text="Processing neural connections..." />
        
        <!-- Loading message -->
        <div class="text-center">
          <p class="text-neuron-text-primary font-heading font-medium">{{ title }}</p>
          <p v-if="message" class="text-neuron-text-secondary text-sm mt-1">{{ message }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import SynapseLoader from './SynapseLoader.vue'

defineProps({
  title: {
    type: String,
    default: "Loading...",
  },
  message: {
    type: String,
    default: "",
  },
});
</script>

<style scoped>
/* Loading overlay animation */
.loading-indicator {
  animation: fade-in 0.3s ease-out;
}

.loading-content {
  animation: scale-in 0.4s ease-out;
  max-width: 400px;
  min-width: 300px;
}

/* Fade in animation */
@keyframes fade-in {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

/* Scale in animation */
@keyframes scale-in {
  from {
    transform: scale(0.9);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

/* Responsive adjustments */
@media (max-width: 640px) {
  .loading-content {
    margin: 1rem;
    min-width: auto;
    max-width: none;
  }
}
</style>
