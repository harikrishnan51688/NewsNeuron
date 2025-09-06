<template>
  <div 
    v-if="show" 
    class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
    @click.self="$emit('close')"
  >
    <div 
      class="relative w-full max-w-4xl max-h-[90vh] bg-dark-card border border-dark-border rounded-2xl shadow-2xl overflow-hidden flex flex-col"
      @click.stop
    >
      <!-- Header -->
      <div class="flex items-start justify-between p-6 border-b border-dark-border">
        <div class="flex-1 pr-4">
          <h2 class="text-xl font-bold text-text-primary mb-2 leading-tight">
            {{ article.title }}
          </h2>
          <div class="flex items-center space-x-4 text-sm text-text-muted">
            <span class="flex items-center space-x-1">
              <div 
                :class="credibilityDotClass"
                class="w-2 h-2 rounded-full"
              ></div>
              <span>{{ 
                typeof article.source === 'string' ? article.source : 
                article.source?.name || 'Unknown Source' 
              }}</span>
            </span>
            <span v-if="article.published_at">
              {{ formatTimeAgo(article.published_at) }}
            </span>
          </div>
        </div>
        
        <button
          @click="$emit('close')"
          class="p-2 text-text-muted hover:text-text-primary hover:bg-dark-border/50 rounded-lg transition-colors"
        >
          <XIcon class="w-5 h-5" />
        </button>
      </div>

      <!-- Content -->
      <div class="overflow-y-auto flex-1 min-h-0">
        <!-- Loading State -->
        <div v-if="isLoading" class="flex flex-col items-center justify-center py-20">
          <SynapseLoader size="lg" />
          <p class="text-text-muted mt-4">Analyzing article...</p>
        </div>

        <!-- Error State -->
        <div v-else-if="error" class="p-6 text-center">
          <div class="text-red-400 mb-4">
            <AlertCircleIcon class="w-12 h-12 mx-auto mb-4" />
            <p class="text-lg font-medium">Analysis Failed</p>
            <p class="text-sm text-text-muted mt-2">{{ error }}</p>
          </div>
          <button
            @click="analyzeArticle"
            class="bg-neuron-glow hover:bg-neuron-glow/80 text-white px-6 py-2 rounded-lg transition-colors"
          >
            Try Again
          </button>
        </div>

        <!-- Analysis Content -->
        <div v-else-if="analysis" class="p-6 space-y-6">
          <!-- Summary Section -->
          <div class="bg-dark-bg/50 rounded-xl p-5 border border-dark-border/50">
            <div class="flex items-center space-x-2 mb-3">
              <FileTextIcon class="w-5 h-5 text-neuron-glow" />
              <h3 class="text-lg font-semibold text-text-primary">Summary</h3>
            </div>
            <p class="text-text-secondary leading-relaxed">
              {{ cleanText(analysis.summary) }}
            </p>
          </div>

          <!-- Credibility & Bias Analysis -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <!-- Credibility Analysis -->
            <div class="bg-dark-bg/50 rounded-xl p-5 border border-dark-border/50">
              <div class="flex items-center space-x-2 mb-3">
                <ShieldCheckIcon class="w-5 h-5 text-blue-400" />
                <h3 class="text-lg font-semibold text-text-primary">Credibility</h3>
              </div>
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-text-muted text-sm">Reliability Score</span>
                  <div class="flex items-center space-x-2">
                    <div class="w-20 h-2 bg-dark-border rounded-full overflow-hidden">
                      <div 
                        class="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                        :style="{ width: (analysis.credibility_analysis.score * 100) + '%' }"
                      ></div>
                    </div>
                    <span class="text-text-primary text-sm font-medium">
                      {{ Math.round(analysis.credibility_analysis.score * 100) }}%
                    </span>
                  </div>
                </div>
                
                <div class="flex items-center justify-between">
                  <span class="text-text-muted text-sm">Tier</span>
                  <span 
                    :class="credibilityBadgeClass"
                    class="px-2 py-1 rounded text-xs font-medium"
                  >
                    {{ credibilityLabel }}
                  </span>
                </div>
                
                <p class="text-text-muted text-xs mt-2">
                  Based on journalism standards, circulation data, and industry ratings
                </p>
              </div>
            </div>

            <!-- Bias Analysis -->
            <div class="bg-dark-bg/50 rounded-xl p-5 border border-dark-border/50">
              <div class="flex items-center space-x-2 mb-3">
                <BalanceIcon class="w-5 h-5 text-purple-400" />
                <h3 class="text-lg font-semibold text-text-primary">Bias Assessment</h3>
              </div>
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-text-muted text-sm">Political Leaning</span>
                  <span 
                    :class="politicalLeaningBadgeClass"
                    class="px-2 py-1 rounded text-xs font-medium"
                  >
                    {{ politicalLeaningLabel }}
                  </span>
                </div>
                
                <div class="flex items-center justify-between">
                  <span class="text-text-muted text-sm">Bias Level</span>
                  <span 
                    :class="biasLevelClass"
                    class="px-2 py-1 rounded text-xs font-medium"
                  >
                    {{ analysis.bias_assessment.bias_level }}
                  </span>
                </div>
                
                <p class="text-text-muted text-xs mt-2">
                  Based on Pew Research studies and media bias analysis
                </p>
              </div>
            </div>
          </div>

          <!-- LLM Interpretation -->
          <div class="bg-gradient-to-br from-neuron-glow/10 to-purple-500/10 rounded-xl p-5 border border-neuron-glow/20">
            <div class="flex items-center space-x-2 mb-3">
              <BrainIcon class="w-5 h-5 text-neuron-glow" />
              <h3 class="text-lg font-semibold text-text-primary">AI Interpretation</h3>
            </div>
            <p class="text-text-secondary leading-relaxed">
              {{ cleanText(analysis.llm_interpretation) }}
            </p>
          </div>

          <!-- Related Articles -->
          <div class="space-y-4">
            <div class="flex items-center space-x-2">
              <LinkIcon class="w-5 h-5 text-orange-400" />
              <h3 class="text-lg font-semibold text-text-primary">Related Articles from Other Sources</h3>
            </div>
            
            <div v-if="analysis.related_articles && analysis.related_articles.length > 0" class="grid grid-cols-1 gap-3">
              <div 
                v-for="relatedArticle in analysis.related_articles"
                :key="relatedArticle.id"
                class="bg-dark-bg/30 rounded-lg p-4 border border-dark-border/50 hover:border-neuron-glow/30 transition-all cursor-pointer group"
                @click="openExternalArticle(relatedArticle.url)"
              >
                <div class="flex items-start justify-between">
                  <div class="flex-1 pr-4">
                    <h4 class="text-text-primary font-medium mb-1 group-hover:text-neuron-glow transition-colors line-clamp-2">
                      {{ relatedArticle.title }}
                    </h4>
                    <p class="text-text-secondary text-sm mb-2 line-clamp-2">
                      {{ relatedArticle.summary }}
                    </p>
                    <div class="flex items-center space-x-3 text-xs text-text-muted">
                      <div class="flex items-center space-x-1">
                        <div 
                          :class="getCredibilityDotClass(relatedArticle.source.credibility)"
                          class="w-1.5 h-1.5 rounded-full"
                        ></div>
                        <span>{{ relatedArticle.source.name }}</span>
                      </div>
                      <span>{{ formatTimeAgo(relatedArticle.published_date) }}</span>
                      <span class="text-neuron-glow">{{ Math.round(relatedArticle.similarity_score * 100) }}% match</span>
                    </div>
                  </div>
                  <ExternalLinkIcon class="w-4 h-4 text-text-muted group-hover:text-neuron-glow transition-colors flex-shrink-0" />
                </div>
              </div>
            </div>
            
            <!-- No Related Articles Message -->
            <div v-else class="bg-dark-bg/30 rounded-lg p-6 border border-dark-border/50 text-center">
              <p class="text-text-muted text-sm">
                No related articles found from other sources.
              </p>
            </div>
          </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="border-t border-dark-border p-4 bg-dark-bg/50 flex-shrink-0">
        <div class="flex items-center justify-between">
          <span class="text-text-muted text-xs">
            Analysis powered by NewsNeuron
          </span>
          <div class="flex space-x-2">
            <button
              @click="openExternalArticle(article.url)"
              class="bg-neuron-glow hover:bg-neuron-glow/80 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-2"
            >
              <span>Read Full Article</span>
              <ExternalLinkIcon class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { 
  XIcon, 
  FileTextIcon, 
  ShieldCheckIcon, 
  LinkIcon, 
  ExternalLinkIcon, 
  AlertCircleIcon,
  BrainIcon
} from 'lucide-vue-next'

// Components
import SynapseLoader from '@/components/ui/SynapseLoader.vue'

// API
import { articlesAPI } from '@/services/api.js'

// Create a Balance icon component since it's not in lucide-vue-next
const BalanceIcon = {
  template: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="6" height="6"/><path d="m21 12-6-6v18l6-6"/><path d="m3 12 6-6v18l-6-6"/></svg>`
}

// Props and Emits
const props = defineProps({
  show: Boolean,
  article: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['close'])

// State
const isLoading = ref(false)
const error = ref('')
const analysis = ref(null)

// Computed properties for credibility display
const credibilityTier = computed(() => {
  return analysis.value?.credibility_analysis?.tier || 'unrated'
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

// Political leaning computed properties
const politicalLeaning = computed(() => {
  return analysis.value?.bias_assessment?.political_leaning || 'center'
})

const politicalLeaningLabel = computed(() => {
  switch (politicalLeaning.value) {
    case 'left':
      return 'Left-leaning'
    case 'right':
      return 'Right-leaning'
    case 'center':
    default:
      return 'Center/Neutral'
  }
})

const politicalLeaningBadgeClass = computed(() => {
  switch (politicalLeaning.value) {
    case 'left':
      return 'bg-blue-600/90 text-white'
    case 'right':
      return 'bg-red-600/90 text-white'
    case 'center':
    default:
      return 'bg-purple-600/90 text-white'
  }
})

const biasLevelClass = computed(() => {
  const level = analysis.value?.bias_assessment?.bias_level || 'Unknown'
  switch (level.toLowerCase()) {
    case 'low':
      return 'bg-green-600/90 text-white'
    case 'moderate':
      return 'bg-yellow-600/90 text-white'
    case 'high':
      return 'bg-red-600/90 text-white'
    default:
      return 'bg-gray-600/90 text-white'
  }
})

// Methods
const analyzeArticle = async () => {
  if (!props.article) return
  
  try {
    isLoading.value = true
    error.value = ''
    
    const response = await articlesAPI.analyzeArticle({
      url: props.article.url,
      title: props.article.title,
      description: props.article.description || '',
      source: props.article.source?.name || props.article.source || '',
      published_at: props.article.published_at || ''
    })
    
    if (response.data.success) {
      analysis.value = response.data
    } else {
      error.value = response.data.error || 'Failed to analyze article'
    }
    
  } catch (err) {
    console.error('Article analysis error:', err)
    error.value = err.response?.data?.error || err.message || 'An unexpected error occurred'
  } finally {
    isLoading.value = false
  }
}

const formatTimeAgo = (dateString) => {
  if (!dateString) return ''
  
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

const getCredibilityDotClass = (credibility) => {
  const tier = credibility?.tier || 'unrated'
  switch (tier) {
    case 'tier1':
      return 'bg-emerald-500'
    case 'tier2':
      return 'bg-blue-500'
    case 'tier3':
      return 'bg-yellow-500'
    default:
      return 'bg-gray-500'
  }
}

const openExternalArticle = (url) => {
  if (url) {
    window.open(url, '_blank', 'noopener,noreferrer')
  }
}

const cleanText = (text) => {
  // Handle non-string data types
  if (!text) return ''
  if (typeof text === 'object') {
    console.warn('Object passed to cleanText:', text)
    return ''
  }
  if (typeof text !== 'string') {
    return String(text)
  }
  
  // Remove any remaining markdown formatting
  let cleaned = text
    .replace(/\*\*([^*]+)\*\*/g, '$1')  // Bold
    .replace(/\*([^*]+)\*/g, '$1')      // Italic
    .replace(/#{1,6}\s+/g, '')          // Headers
    .replace(/^\s*-\s+/gm, '')          // Bullet points
    .replace(/^\s*\d+\.\s+/gm, '')      // Numbered lists
    .replace(/`([^`]+)`/g, '$1')        // Code blocks
    .replace(/\n{2,}/g, ' ')            // Multiple newlines
    .replace(/\s+/g, ' ')               // Multiple spaces
    .trim()
  
  return cleaned
}

// Watch for modal show/hide and article changes
watch(() => props.show, (newShow) => {
  if (newShow && props.article) {
    // Always reset and analyze when modal opens
    analysis.value = null
    error.value = ''
    analyzeArticle()
  }
})

// Watch for article changes while modal is open
watch(() => props.article, (newArticle, oldArticle) => {
  if (props.show && newArticle && newArticle !== oldArticle) {
    // Reset and analyze when article changes
    analysis.value = null
    error.value = ''
    analyzeArticle()
  }
})

// Handle escape key
const handleEscape = (event) => {
  if (event.key === 'Escape' && props.show) {
    emit('close')
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleEscape)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleEscape)
})
</script>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Custom scrollbar */
.overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}

.overflow-y-auto::-webkit-scrollbar-track {
  background: var(--neuron-bg-content);
}

.overflow-y-auto::-webkit-scrollbar-thumb {
  background: var(--neuron-border-secondary);
  border-radius: 3px;
}

.overflow-y-auto::-webkit-scrollbar-thumb:hover {
  background: var(--neuron-glow);
}
</style>
