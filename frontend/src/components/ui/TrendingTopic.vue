<template>
  <div
    class="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg transition-colors duration-200 cursor-pointer"
    @click="handleClick"
  >
    <div class="flex-1">
      <h4 class="text-heading text-neuron-text-primary text-sm">{{ topic.topic }}</h4>
      <p class="text-xs text-neuron-text-secondary mt-1 font-body">
        {{ topic.mention_count }} mentions
      </p>
    </div>

    <div class="flex items-center space-x-2">
      <div class="flex items-center">
        <div
          class="w-16 h-2 bg-gray-200 rounded-full overflow-hidden"
          :title="`Trend score: ${Math.round(topic.trend_score * 100)}%`"
        >
          <div
            class="h-full transition-all duration-300"
            :class="trendColorClass"
            :style="{ width: `${topic.trend_score * 100}%` }"
          ></div>
        </div>
      </div>

      <ChevronRightIcon class="w-4 h-4 text-gray-400" />
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";
import { useRouter } from "vue-router";
import { ChevronRightIcon } from "@heroicons/vue/24/outline";

const props = defineProps({
  topic: {
    type: Object,
    required: true,
  },
});

const router = useRouter();

const trendColorClass = computed(() => {
  const score = props.topic.trend_score;
  if (score >= 0.8) return "bg-success-500";
  if (score >= 0.6) return "bg-warning-500";
  return "bg-primary-500";
});

function handleClick() {
  router.push(`/search?q=${encodeURIComponent(props.topic.topic)}`);
}
</script>
