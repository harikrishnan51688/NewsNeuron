<template>
  <Teleport to="body">
    <div class="fixed top-6 right-6 z-50 space-y-3 max-w-sm">
      <TransitionGroup name="toast" tag="div">
        <ToastNotification
          v-for="notification in notifications"
          :key="notification.id"
          :notification="notification"
          @close="removeNotification"
        />
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup>
import { computed } from "vue";
import { useAppStore } from "@/stores/app";
import ToastNotification from "./ToastNotification.vue";

const appStore = useAppStore();

const notifications = computed(() => appStore.notifications);

function removeNotification(id) {
  appStore.removeNotification(id);
}
</script>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100%) scale(0.95);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100%) scale(0.95);
}

.toast-move {
  transition: transform 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
}
</style>
