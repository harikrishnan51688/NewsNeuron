<template>
  <div class="typing-indicator flex items-center space-x-1">
    <div class="flex space-x-1">
      <div
        v-for="i in 3"
        :key="i"
        class="w-2 h-2 bg-neuron-glow rounded-full animate-pulse"
        :style="{ animationDelay: `${(i - 1) * 0.2}s` }"
      ></div>
    </div>
    <span class="text-sm text-neuron-text-secondary ml-2">Neural Assistant is thinking...</span>
  </div>
</template>

<script setup>
// No props needed for this simple component
</script>

<style scoped>
@keyframes pulse {
  0%, 80%, 100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.animate-pulse {
  animation: pulse 1.4s ease-in-out infinite both;
}
</style>
