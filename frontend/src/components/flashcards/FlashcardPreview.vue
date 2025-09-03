<template>
  <article 
    class="flashcard-preview card-neuron cursor-pointer group relative overflow-hidden"
    @click="$emit('click', flashcard)"
    :style="{ height: cardHeight }"
  >
    
    <!-- Category Badge -->
    <div class="absolute top-4 left-4 z-10">
      <span 
        class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium"
        :class="categoryClasses"
      >
        <component :is="categoryIcon" class="w-3 h-3 mr-1" />
        {{ formatCategory(flashcard.category) }}
      </span>
    </div>
    
    <!-- Actions Menu -->
    <div class="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
      <div class="flex items-center space-x-2">
        
        <!-- Bookmark -->
        <button
          @click.stop="$emit('bookmark', flashcard)"
          class="btn-icon w-8 h-8 bg-neuron-bg-primary/80 backdrop-blur-sm"
          :class="{ 'text-neuron-glow': flashcard.bookmarked }"
          title="Bookmark"
        >
          <Bookmark class="w-4 h-4" :class="{ 'fill-current': flashcard.bookmarked }" />
        </button>
        
        <!-- Share -->
        <button
          @click.stop="$emit('share', flashcard)"
          class="btn-icon w-8 h-8 bg-neuron-bg-primary/80 backdrop-blur-sm"
          title="Share"
        >
          <Share2 class="w-4 h-4" />
        </button>
        
        <!-- More Options -->
        <button
          @click.stop="showOptions = !showOptions"
          class="btn-icon w-8 h-8 bg-neuron-bg-primary/80 backdrop-blur-sm"
          title="More options"
        >
          <MoreVertical class="w-4 h-4" />
        </button>
        
      </div>
      
      <!-- Options Dropdown -->
      <div v-if="showOptions" class="absolute right-0 top-full mt-2 w-48 bg-neuron-bg-content border border-neuron-border rounded-lg shadow-neuron-lg py-2 z-20">
        <button class="w-full px-4 py-2 text-left text-sm text-neuron-text-secondary hover:text-neuron-text-primary hover:bg-neuron-bg-primary transition-colors">
          <ExternalLink class="w-4 h-4 mr-2 inline" />
          Open Source
        </button>
        <button class="w-full px-4 py-2 text-left text-sm text-neuron-text-secondary hover:text-neuron-text-primary hover:bg-neuron-bg-primary transition-colors">
          <Eye class="w-4 h-4 mr-2 inline" />
          View Timeline
        </button>
        <button class="w-full px-4 py-2 text-left text-sm text-neuron-text-secondary hover:text-neuron-text-primary hover:bg-neuron-bg-primary transition-colors">
          <Flag class="w-4 h-4 mr-2 inline" />
          Report Issue
        </button>
      </div>
    </div>
    
    <!-- Main Content -->
    <div class="p-6 h-full flex flex-col">
      
      <!-- Title -->
      <h3 class="text-lg font-heading font-medium text-neuron-text-primary mb-3 line-clamp-2 group-hover:text-neuron-glow transition-colors duration-200">
        {{ flashcard.title }}
      </h3>
      
      <!-- Summary -->
      <p class="text-neuron-text-secondary text-sm leading-relaxed mb-4 flex-1 line-clamp-4">
        {{ flashcard.summary }}
      </p>
      
      <!-- Connections Indicator -->
      <div v-if="flashcard.connections && flashcard.connections.length > 0" class="mb-4">
        <div class="flex items-center space-x-2">
          <div class="flex -space-x-1">
            <div 
              v-for="(connection, index) in flashcard.connections.slice(0, 3)"
              :key="connection"
              :class="getConnectionColor(index)"
              class="w-6 h-6 rounded-full border-2 border-neuron-bg-content flex items-center justify-center"
              :style="{ zIndex: 10 - index }"
            >
              <div :class="getConnectionDotColor(index)" class="w-2 h-2 rounded-full animate-pulse"></div>
            </div>
          </div>
          <span class="text-xs text-neuron-text-secondary">
            {{ flashcard.connections.length }} {{ flashcard.connections.length === 1 ? 'connection' : 'connections' }}
          </span>
        </div>
      </div>
      
      <!-- Footer -->
      <div class="flex items-center justify-between pt-4 border-t border-neuron-border">
        
        <!-- Source & Reading Time -->
        <div class="flex items-center space-x-3 text-xs text-neuron-text-secondary">
          <span class="font-medium">{{ flashcard.source }}</span>
          <div class="w-1 h-1 bg-neuron-border rounded-full"></div>
          <span>{{ flashcard.reading_time || '2 min read' }}</span>
        </div>
        
        <!-- Timestamp -->
        <time 
          :datetime="flashcard.created_at"
          class="text-xs text-neuron-text-secondary"
          :title="formatFullDate(flashcard.created_at)"
        >
          {{ formatRelativeTime(flashcard.created_at) }}
        </time>
        
      </div>
      
    </div>
    
    <!-- Hover Effect Overlay -->
    <div class="absolute inset-0 bg-gradient-to-t from-neuron-glow/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
    
    <!-- Connection Lines (when hovering) -->
    <div v-if="showConnections" class="absolute inset-0 pointer-events-none">
      <svg class="w-full h-full">
        <line
          v-for="line in connectionLines"
          :key="line.id"
          :x1="line.x1" :y1="line.y1"
          :x2="line.x2" :y2="line.y2"
          stroke="var(--neuron-glow)"
          stroke-width="2"
          stroke-dasharray="5,5"
          class="synapse-line opacity-60"
        />
      </svg>
    </div>
    
  </article>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { 
  Bookmark, 
  Share2, 
  MoreVertical, 
  ExternalLink, 
  Eye, 
  Flag,
  Cpu,
  Briefcase,
  Users,
  FlaskConical,
  Globe
} from 'lucide-vue-next'

// Props
const props = defineProps({
  flashcard: {
    type: Object,
    required: true
  },
  showConnections: {
    type: Boolean,
    default: false
  }
})

// Emits
defineEmits(['click', 'bookmark', 'share'])

// Reactive state
const showOptions = ref(false)

// Computed
const cardHeight = computed(() => {
  // Dynamic height based on content length
  const baseHeight = 280
  const titleLength = props.flashcard.title?.length || 0
  const summaryLength = props.flashcard.summary?.length || 0
  
  // Add height for longer content
  let extraHeight = 0
  if (titleLength > 60) extraHeight += 20
  if (summaryLength > 150) extraHeight += 40
  if (props.flashcard.connections?.length > 0) extraHeight += 20
  
  return `${baseHeight + extraHeight}px`
})

const categoryClasses = computed(() => {
  const category = props.flashcard.category?.toLowerCase()
  switch (category) {
    case 'technology':
      return 'bg-accent-violet/20 text-accent-violet border border-accent-violet/30'
    case 'business':
      return 'bg-accent-emerald/20 text-accent-emerald border border-accent-emerald/30'
    case 'politics':
      return 'bg-accent-rose/20 text-accent-rose border border-accent-rose/30'
    case 'science':
      return 'bg-accent-amber/20 text-accent-amber border border-accent-amber/30'
    case 'environment':
      return 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
    case 'economics':
      return 'bg-accent-orange/20 text-accent-orange border border-accent-orange/30'
    default:
      return 'bg-neuron-border text-neuron-text-secondary border border-neuron-border'
  }
})

const categoryIcon = computed(() => {
  const category = props.flashcard.category?.toLowerCase()
  switch (category) {
    case 'technology':
      return Cpu
    case 'business':
      return Briefcase
    case 'politics':
      return Users
    case 'science':
      return FlaskConical
    default:
      return Globe
  }
})

const connectionLines = computed(() => {
  // Generate random connection lines for visual effect
  if (!props.showConnections || !props.flashcard.connections?.length) return []
  
  return props.flashcard.connections.slice(0, 3).map((_, index) => ({
    id: index,
    x1: Math.random() * 100 + '%',
    y1: Math.random() * 100 + '%',
    x2: Math.random() * 100 + '%',
    y2: Math.random() * 100 + '%'
  }))
})

// Methods
const formatCategory = (category) => {
  return category ? category.charAt(0).toUpperCase() + category.slice(1) : 'General'
}

const formatRelativeTime = (dateString) => {
  const date = new Date(dateString)
  const now = new Date()
  const diff = now - date
  
  const minutes = Math.floor(diff / (1000 * 60))
  const hours = Math.floor(diff / (1000 * 60 * 60))
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days < 7) return `${days}d ago`
  
  return date.toLocaleDateString()
}

const formatFullDate = (dateString) => {
  return new Date(dateString).toLocaleString()
}

// Connection color variety
const getConnectionColor = (index) => {
  const colors = [
    'bg-accent-emerald/20',
    'bg-accent-violet/20', 
    'bg-accent-amber/20'
  ]
  return colors[index % colors.length]
}

const getConnectionDotColor = (index) => {
  const colors = [
    'bg-accent-emerald',
    'bg-accent-violet',
    'bg-accent-amber'
  ]
  return colors[index % colors.length]
}

// Close options when clicking outside
const handleClickOutside = (event) => {
  if (!event.target.closest('.flashcard-preview')) {
    showOptions.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
/* Line clamp utilities */
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.line-clamp-4 {
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Enhanced hover effects */
.flashcard-preview {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.flashcard-preview:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 167, 225, 0.2);
}

/* Connection lines animation */
.synapse-line {
  animation: synapse-flow 3s ease-in-out infinite;
}

@keyframes synapse-flow {
  0%, 100% { stroke-dashoffset: 0; }
  50% { stroke-dashoffset: 10; }
}

/* Focus states for accessibility */
.flashcard-preview:focus-visible {
  outline: 2px solid var(--neuron-glow);
  outline-offset: 2px;
}

/* Smooth transitions for all interactive elements */
button {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Enhanced backdrop blur for action buttons */
.btn-icon {
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}
</style>
