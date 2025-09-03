<template>
  <div
    :class="[
      'bg-dark-card border border-dark-border rounded-2xl p-4 max-w-sm cursor-pointer shadow-2xl backdrop-blur-sm',
      'transform transition-all duration-300 hover:scale-[1.02]',
      typeClasses[notification.type],
      { 'opacity-75': notification.read }
    ]"
    @click="handleClick"
  >
    <div class="flex items-start space-x-3">
      <!-- Icon -->
      <div class="flex-shrink-0">
        <div :class="['w-8 h-8 rounded-lg flex items-center justify-center', typeIconBgClasses[notification.type]]">
          <component
            :is="typeIcon[notification.type]"
            :class="['w-4 h-4', typeIconClasses[notification.type]]"
          />
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 min-w-0">
        <div class="flex items-start justify-between">
          <div class="flex-1 min-w-0">
            <p v-if="notification.title" class="font-semibold text-text-primary text-sm">
              {{ notification.title }}
            </p>
            <p
              class="text-text-secondary text-sm leading-relaxed"
              :class="{ 'mt-1': notification.title }"
            >
              {{ notification.message }}
            </p>
          </div>

          <!-- Close Button -->
          <button
            @click.stop="$emit('close', notification.id)"
            class="ml-2 p-1 rounded-lg text-text-muted hover:text-text-secondary hover:bg-dark-background transition-all"
          >
            <XIcon class="w-4 h-4" />
          </button>
        </div>

        <!-- Actions -->
        <div v-if="notification.actions" class="flex items-center mt-3 space-x-2">
          <button
            v-for="action in notification.actions"
            :key="action.id"
            :class="[
              'px-3 py-1 rounded-lg text-xs font-medium transition-all',
              action.primary
                ? 'bg-neuron-glow text-white hover:bg-neuron-glow/80'
                : 'bg-dark-background border border-dark-border text-text-secondary hover:text-text-primary hover:border-neuron-glow/50'
            ]"
            @click.stop="handleAction(action)"
          >
            {{ action.label }}
          </button>
        </div>

        <!-- Timestamp -->
        <p v-if="notification.timestamp" class="text-text-muted text-xs mt-2">
          {{ formatTime(notification.timestamp) }}
        </p>
      </div>
    </div>

    <!-- Progress bar for auto-hide -->
    <div
      v-if="notification.autoHide && showProgress"
      class="mt-3 h-1 bg-dark-background rounded-full overflow-hidden"
    >
      <div
        class="h-full transition-all ease-linear rounded-full"
        :class="typeProgressClasses[notification.type]"
        :style="{ width: `${progress}%` }"
      ></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon as InfoIcon,
  XMarkIcon as XIcon,
  SparklesIcon
} from "@heroicons/vue/24/outline";

const props = defineProps({
  notification: {
    type: Object,
    required: true,
  },
});

const emit = defineEmits(["close", "action"]);

// State
const progress = ref(100);
const showProgress = ref(true);
let progressInterval = null;

// Type configurations
const typeClasses = {
  success: "border-green-500/30 shadow-green-500/10",
  error: "border-red-500/30 shadow-red-500/10",
  warning: "border-yellow-500/30 shadow-yellow-500/10",
  info: "border-blue-500/30 shadow-blue-500/10",
  ai: "border-neuron-glow/30 shadow-neuron-glow/10",
};

const typeIcon = {
  success: CheckCircleIcon,
  error: XCircleIcon,
  warning: ExclamationTriangleIcon,
  info: InfoIcon,
  ai: SparklesIcon
};

const typeIconClasses = {
  success: "text-green-400",
  error: "text-red-400",
  warning: "text-yellow-400",
  info: "text-blue-400",
  ai: "text-neuron-glow"
};

const typeIconBgClasses = {
  success: "bg-green-500/20",
  error: "bg-red-500/20",
  warning: "bg-yellow-500/20",
  info: "bg-blue-500/20",
  ai: "bg-neuron-glow/20"
};

const typeProgressClasses = {
  success: "bg-green-400",
  error: "bg-red-400",
  warning: "bg-yellow-400",
  info: "bg-blue-400",
  ai: "bg-neuron-glow"
};

// Methods
function handleClick() {
  if (props.notification.clickAction) {
    emit("action", props.notification.clickAction);
  }
}

function handleAction(action) {
  emit("action", action);
}

function formatTime(timestamp) {
  const now = new Date();
  const diff = now - timestamp;
  const minutes = Math.floor(diff / (1000 * 60));
  
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Auto-hide functionality
onMounted(() => {
  if (props.notification.autoHide) {
    const duration = props.notification.duration || 5000;
    const interval = 50;
    const step = (100 / duration) * interval;

    progressInterval = setInterval(() => {
      progress.value -= step;
      
      if (progress.value <= 0) {
        emit("close", props.notification.id);
      }
    }, interval);
  }
});

onUnmounted(() => {
  if (progressInterval) {
    clearInterval(progressInterval);
  }
});
</script>

const typeProgressClasses = {
  success: "bg-success-400",
  error: "bg-error-400",
  warning: "bg-warning-400",
  info: "bg-primary-400",

