<template>
  <div 
    class="bg-dark-card border border-dark-border rounded-2xl p-6 hover:border-neuron-glow/30 transition-all duration-300 cursor-pointer group"
    @click="openSummary"
  >
    <!-- Article Image -->
    <div v-if="headline.image" class="relative mb-4">
      <img 
        :src="headline.image" 
        :alt="headline.title"
        class="w-full h-48 object-cover rounded-xl"
        @error="handleImageError"
      />
      <!-- Source Badges with improved readability -->
      <div class="absolute top-3 right-3 flex flex-col space-y-1 items-end">
        <!-- Credibility Badge -->
        <span 
          :class="credibilityBadgeClass"
          class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-sm border border-white/20 shadow-lg"
        >
          {{ credibilityLabel }}
        </span>
        <!-- Political Leaning Badge -->
        <span 
          :class="politicalLeaningBadgeClass"
          class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-sm border border-white/20 shadow-lg"
        >
          {{ politicalLeaningLabel }}
        </span>
      </div>
    </div>

    <!-- Article Content -->
    <div class="space-y-3">
      <!-- Category Tag -->
      <div class="flex items-center justify-between">
        <span class="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-neuron-glow/20 text-neuron-glow">
          {{ headline.category }}
        </span>
        <span class="text-text-muted text-xs">
          {{ formatTimeAgo(headline.published_at) }}
        </span>
      </div>

      <!-- Title -->
      <h3 class="text-text-primary font-semibold text-lg leading-tight group-hover:text-neuron-glow transition-colors">
        {{ headline.title }}
      </h3>

      <!-- Description -->
      <p class="text-text-secondary text-sm leading-relaxed line-clamp-3">
        {{ headline.description }}
      </p>

      <!-- Source Info -->
      <div class="flex items-center justify-between pt-2 border-t border-dark-border">
        <div class="flex items-center space-x-2">
          <div class="flex items-center space-x-1">
            <div 
              :class="credibilityDotClass"
              class="w-2 h-2 rounded-full"
            ></div>
            <span class="text-text-muted text-xs font-medium">
              {{ headline.source.name }}
            </span>
          </div>
        </div>
        
        <div class="flex items-center space-x-2">
          <button
            @click.stop="openSummary"
            class="text-neuron-glow hover:text-neuron-glow/80 text-xs font-medium flex items-center space-x-1 transition-colors"
          >
            <span>View Summary</span>
            <FileTextIcon class="w-3 h-3" />
          </button>
          
          <button
            @click.stop="openArticle"
            class="text-text-muted hover:text-neuron-glow text-xs font-medium flex items-center space-x-1 transition-colors"
          >
            <span>Read more</span>
            <ExternalLinkIcon class="w-3 h-3" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { ExternalLinkIcon, FileTextIcon } from 'lucide-vue-next'

// Props
const props = defineProps({
  headline: {
    type: Object,
    required: true
  }
})

// Emits
const emit = defineEmits(['openSummary'])

// Computed properties for credibility display
const credibilityTier = computed(() => {
  return props.headline.source.credibility?.tier || 'unrated'
})

const credibilityLabel = computed(() => {
  switch (credibilityTier.value) {
    case 'tier1':
      return 'Highly Credible'
    case 'tier2':
      return 'Credible'
    case 'tier3':
      return 'Specialized'
    default:
      return 'Unrated'
  }
})

// Computed properties for political leaning display
const politicalLeaning = computed(() => {
  return props.headline.source.credibility?.political_leaning || 'center'
})

const politicalLeaningLabel = computed(() => {
  switch (politicalLeaning.value) {
    case 'left':
      return 'Left'
    case 'right':
      return 'Right'
    case 'center':
    default:
      return 'Center'
  }
})

const credibilityBadgeClass = computed(() => {
  switch (credibilityTier.value) {
    case 'tier1':
      return 'bg-emerald-600/90 text-white'
    case 'tier2':
      return 'bg-blue-600/90 text-white'
    case 'tier3':
      return 'bg-yellow-600/90 text-white'
    default:
      return 'bg-gray-600/90 text-white'
  }
})

const politicalLeaningBadgeClass = computed(() => {
  switch (politicalLeaning.value) {
    case 'left':
      return 'bg-blue-700/90 text-white'
    case 'right':
      return 'bg-red-700/90 text-white'
    case 'center':
    default:
      return 'bg-purple-700/90 text-white'
  }
})

const credibilityDotClass = computed(() => {
  switch (credibilityTier.value) {
    case 'tier1':
      return 'bg-emerald-500'
    case 'tier2':
      return 'bg-blue-500'
    case 'tier3':
      return 'bg-yellow-500'
    default:
      return 'bg-gray-500'
  }
})

// Methods
const formatTimeAgo = (dateString) => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now - date
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffHours / 24)
  
  if (diffHours < 1) return 'Just now'
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric' 
  })
}

const handleImageError = (event) => {
  // Hide image if it fails to load
  event.target.style.display = 'none'
}

const openArticle = () => {
  if (props.headline.url) {
    window.open(props.headline.url, '_blank', 'noopener,noreferrer')
  }
}

const openSummary = () => {
  emit('openSummary', props.headline)
}
</script>

<style scoped>
.line-clamp-3 {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
