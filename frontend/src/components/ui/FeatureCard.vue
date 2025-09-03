<template>
  <div class="card hover:shadow-lg transition-all duration-300 group">
    <div class="card-body text-center">
      <div
        class="w-12 h-12 mx-auto mb-4 rounded-lg flex items-center justify-center"
        :class="iconBgClass"
      >
        <component :is="feature.icon" class="w-6 h-6" :class="iconClass" />
      </div>

      <h3
        class="text-lg text-heading text-neuron-text-primary mb-2 group-hover:text-neuron-glow transition-colors duration-200"
      >
        {{ feature.title }}
      </h3>

      <p class="text-neuron-text-secondary font-body">
        {{ feature.description }}
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  feature: {
    type: Object,
    required: true,
  },
});

const iconBgClass = computed(() => {
  const colorMap = {
    primary: "bg-primary-100",
    accent: "bg-accent-100",
    success: "bg-success-100",
    warning: "bg-warning-100",
    error: "bg-error-100",
  };
  return colorMap[props.feature.color] || "bg-primary-100";
});

const iconClass = computed(() => {
  const colorMap = {
    primary: "text-primary-600",
    accent: "text-accent-600",
    success: "text-success-600",
    warning: "text-warning-600",
    error: "text-error-600",
  };
  return colorMap[props.feature.color] || "text-primary-600";
});
</script>
