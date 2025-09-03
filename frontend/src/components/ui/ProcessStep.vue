<template>
  <div class="text-center">
    <div class="relative">
      <!-- Step Number -->
      <div
        class="w-12 h-12 mx-auto mb-4 bg-neuron-glow text-neuron-bg-primary rounded-full flex items-center justify-center font-bold text-lg font-heading"
      >
        {{ number }}
      </div>

      <!-- Step Icon -->
      <div
        class="absolute -top-2 -right-2 w-8 h-8 bg-neuron-glow-hover text-neuron-bg-primary rounded-full flex items-center justify-center"
      >
        <component :is="step.icon" class="w-4 h-4" />
      </div>
    </div>

    <h3 class="text-xl font-heading text-neuron-text-primary mb-3">
      {{ step.title }}
    </h3>

    <p class="text-neuron-text-secondary font-body">
      {{ step.description }}
    </p>
  </div>
</template>

<script setup>
defineProps({
  step: {
    type: Object,
    required: true,
  },
  number: {
    type: Number,
    required: true,
  },
});
</script>
