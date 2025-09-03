<template>
  <div 
    class="flashcard-list-item group bg-neuron-bg-content border border-neuron-border rounded-lg p-4 hover:border-neuron-glow/30 hover:shadow-neuron-lg transition-all duration-300 cursor-pointer"
    @click="$emit('click')"
    role="article"
    :aria-label="`News item: ${flashcard.title}`"
  >
    <div class="flex items-start space-x-4">
      
      <!-- Category & Time Badge -->
      <div class="flex-shrink-0 flex flex-col space-y-2">
        <span 
          class="category-badge inline-flex items-center px-2 py-1 rounded-md text-xs font-medium"
          :class="categoryClasses"
        >
          {{ flashcard.category || 'General' }}
        </span>
        <span class="text-xs text-neuron-text-secondary">{{ formattedTime }}</span>
      </div>
      
      <!-- Content -->
      <div class="flex-1 min-w-0">
        <div class="flex items-start justify-between">
          <div class="flex-1">
            <h3 class="text-lg text-heading text-neuron-text-primary group-hover:text-neuron-glow transition-colors duration-300 line-clamp-2">
              {{ flashcard.title }}
            </h3>
            <p class="text-body-sans text-neuron-text-secondary mt-2 line-clamp-2">
              {{ flashcard.summary }}
            </p>
            
            <!-- Metadata -->
            <div class="flex items-center space-x-4 mt-3 text-sm text-neuron-text-secondary">
              <span class="flex items-center">
                <Clock class="w-4 h-4 mr-1" />
                {{ flashcard.reading_time || '2 min read' }}
              </span>
              <span class="flex items-center">
                <ExternalLink class="w-4 h-4 mr-1" />
                {{ flashcard.source || 'NewsNeuron' }}
              </span>
              <span v-if="flashcard.connections?.length" class="flex items-center">
                <Network class="w-4 h-4 mr-1" />
                {{ flashcard.connections.length }} connection{{ flashcard.connections.length !== 1 ? 's' : '' }}
              </span>
            </div>
          </div>
          
          <!-- Actions -->
          <div class="flex items-center space-x-2 ml-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <button
              @click.stop="$emit('bookmark')"
              class="btn-icon p-2"
              :class="{ 'text-neuron-glow': flashcard.bookmarked }"
              :title="flashcard.bookmarked ? 'Remove bookmark' : 'Add bookmark'"
            >
              <Bookmark 
                class="w-4 h-4" 
                :class="flashcard.bookmarked ? 'fill-current' : ''"
              />
            </button>
            <button
              @click.stop="$emit('share')"
              class="btn-icon p-2"
              title="Share"
            >
              <Share2 class="w-4 h-4" />
            </button>
            <button
              @click.stop="openDetails"
              class="btn-icon p-2"
              title="View details"
            >
              <ChevronRight class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { formatDistanceToNow } from 'date-fns'

// Icons
import { 
  Clock, 
  ExternalLink, 
  Network, 
  Bookmark, 
  Share2, 
  ChevronRight 
} from 'lucide-vue-next'

// Props
const props = defineProps({
  flashcard: {
    type: Object,
    required: true
  }
})

// Emits
const emit = defineEmits(['click', 'bookmark', 'share'])

// Computed
const formattedTime = computed(() => {
  try {
    return formatDistanceToNow(new Date(props.flashcard.created_at), { addSuffix: true })
  } catch {
    return 'Recent'
  }
})

const categoryClasses = computed(() => {
  const baseClasses = 'bg-opacity-20 border border-current'
  const categoryColors = {
    technology: 'text-blue-400 bg-blue-400',
    business: 'text-green-400 bg-green-400',
    politics: 'text-purple-400 bg-purple-400',
    science: 'text-cyan-400 bg-cyan-400',
    health: 'text-pink-400 bg-pink-400',
    sports: 'text-orange-400 bg-orange-400',
    entertainment: 'text-yellow-400 bg-yellow-400',
  }
  
  const category = props.flashcard.category?.toLowerCase() || 'general'
  return `${baseClasses} ${categoryColors[category] || 'text-neuron-text-secondary bg-neuron-text-secondary'}`
})

// Methods
const openDetails = () => {
  emit('click')
}
</script>

<style scoped>
/* List item hover effects */
.flashcard-list-item:hover {
  transform: translateY(-1px);
}

/* Category badge styling */
.category-badge {
  text-transform: capitalize;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}

/* Text truncation utilities */
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Smooth transitions */
.btn-icon {
  transition: all 0.2s ease-in-out;
}

.btn-icon:hover {
  transform: scale(1.1);
}

/* Focus states for accessibility */
.flashcard-list-item:focus-visible {
  outline: 2px solid var(--neuron-glow);
  outline-offset: 2px;
}

/* Responsive adjustments */
@media (max-width: 640px) {
  .flashcard-list-item {
    padding: 1rem;
  }
  
  .flex-col {
    flex-direction: row;
    gap: 0.5rem;
  }
}
</style>
