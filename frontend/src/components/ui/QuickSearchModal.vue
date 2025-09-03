<template>
  <Teleport to="body">
    <div v-if="isOpen" class="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
      <div class="bg-dark-card border border-dark-border rounded-2xl w-full max-w-lg p-6">
        <h3 class="text-lg font-semibold text-text-primary mb-4">Search NewsNeuron</h3>
        <input
          v-model="query"
          type="text"
          placeholder="Search news, entities, topics..."
          class="w-full px-4 py-3 bg-dark-background border border-dark-border rounded-xl text-text-primary placeholder-text-secondary focus:ring-2 focus:ring-neuron-glow/30 focus:border-neuron-glow transition-all"
          @keydown.escape="$emit('close')"
          @keydown.enter="handleSearch"
        />
        <div class="flex justify-end space-x-3 mt-4">
          <button
            @click="$emit('close')"
            class="px-4 py-2 text-text-secondary hover:text-text-primary transition-colors"
          >
            Cancel
          </button>
          <button
            @click="handleSearch"
            class="px-4 py-2 bg-neuron-glow text-white rounded-lg hover:bg-neuron-glow/80 transition-colors"
          >
            Search
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  isOpen: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['close', 'search'])
const query = ref('')

const handleSearch = () => {
  if (query.value.trim()) {
    emit('search', { query: query.value })
    emit('close')
  }
}
</script>
