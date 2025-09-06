<template>
  <div class="dashboard-view h-full overflow-y-auto">
    
    <!-- Main Content - Top Headlines -->
    <section class="main-content px-6 py-12">
      <div class="content-width">
        
        <!-- Section Header -->
        <div class="flex items-center justify-between mb-8">
          <div>
            <h2 class="text-2xl font-bold text-text-primary mb-2">Top Headlines</h2>
            <p class="text-text-secondary">Latest breaking news and trending stories</p>
          </div>
          
          <!-- Category Filter -->
          <div class="flex items-center space-x-4">
            <select 
              v-model="selectedCategory" 
              @change="fetchHeadlines"
              class="bg-dark-card border border-dark-border rounded-lg px-4 py-2 text-text-primary focus:border-neuron-glow focus:outline-none"
            >
              <option value="general">General</option>
              <option value="technology">Technology</option>
              <option value="business">Business</option>
              <option value="science">Science</option>
              <option value="health">Health</option>
              <option value="sports">Sports</option>
              <option value="entertainment">Entertainment</option>
            </select>
            
            <button
              @click="fetchHeadlines"
              :disabled="isLoading"
              class="bg-neuron-glow hover:bg-neuron-glow/80 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCwIcon :class="['w-4 h-4', { 'animate-spin': isLoading }]" />
            </button>
          </div>
        </div>
        
        <!-- Loading State -->
        <div v-if="isLoading" class="flex justify-center items-center py-20">
          <SynapseLoader size="lg" show-text loading-text="Loading headlines..." />
        </div>
        
        <!-- Error State -->
        <div v-else-if="error" class="text-center py-20">
          <div class="text-red-400 mb-4">
            <AlertCircleIcon class="w-12 h-12 mx-auto mb-4" />
            <p class="text-lg font-medium">Failed to load headlines</p>
            <p class="text-sm text-text-muted mt-2">{{ error }}</p>
          </div>
          <button
            @click="fetchHeadlines"
            class="bg-neuron-glow hover:bg-neuron-glow/80 text-white px-6 py-2 rounded-lg transition-colors"
          >
            Try Again
          </button>
        </div>
        
        <!-- Headlines Grid -->
        <div v-else-if="headlines.length > 0" class="space-y-6">
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            <TopHeadlineCard
              v-for="headline in displayedHeadlines"
              :key="headline.id"
              :headline="headline"
              class="animate-fade-in"
              @openSummary="openSummaryModal"
            />
          </div>
          
          <!-- Load More Button -->
          <div class="flex justify-center">
            <button
              @click="loadMoreHeadlines"
              :disabled="isLoadingMore"
              class="bg-dark-card hover:bg-dark-border border border-dark-border text-text-primary px-6 py-3 rounded-lg transition-colors disabled:opacity-50 flex items-center space-x-2"
            >
              <RefreshCwIcon :class="['w-4 h-4', { 'animate-spin': isLoadingMore }]" />
              <span>{{ isLoadingMore ? 'Loading More...' : 'Load More Headlines' }}</span>
            </button>
          </div>
        </div>
        
        <!-- Empty State -->
        <div v-else class="text-center py-20">
          <div class="text-text-muted">
            <NewsIcon class="w-12 h-12 mx-auto mb-4" />
            <p class="text-lg font-medium">No headlines available</p>
            <p class="text-sm mt-2">Try selecting a different category</p>
          </div>
        </div>
        
      </div>
    </section>

    <!-- Article Summary Modal -->
    <ArticleSummaryModal
      :show="showSummaryModal"
      :article="selectedArticle"
      @close="closeSummaryModal"
    />

  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { RefreshCwIcon, AlertCircleIcon } from 'lucide-vue-next'

// Components
import TopHeadlineCard from '@/components/headlines/TopHeadlineCard.vue'
import SynapseLoader from '@/components/ui/SynapseLoader.vue'
import ArticleSummaryModal from '@/components/ui/ArticleSummaryModal.vue'

// API
import { headlinesAPI } from '@/services/api.js'

// Icons (using a news icon placeholder - you can replace with actual icon)
const NewsIcon = RefreshCwIcon

// State
const isLoading = ref(true)
const isLoadingMore = ref(false)
const error = ref('')
const headlines = ref([])
const selectedCategory = ref('general')
const currentLimit = ref(4)

// Modal state
const showSummaryModal = ref(false)
const selectedArticle = ref(null)

// Computed - Show all headlines (no limit for this minimalistic approach)
const displayedHeadlines = computed(() => {
  return headlines.value
})

// Methods
const fetchHeadlines = async (reset = true) => {
  try {
    if (reset) {
      isLoading.value = true
      currentLimit.value = 4
    } else {
      isLoadingMore.value = true
    }
    error.value = ''
    
    const response = await headlinesAPI.getTopHeadlines({
      category: selectedCategory.value,
      lang: 'en',
      country: 'us',
      max_articles: currentLimit.value
    })
    
    if (response.data.success) {
      headlines.value = response.data.headlines
    } else {
      error.value = response.data.error || 'Failed to fetch headlines'
    }
    
  } catch (err) {
    console.error('Headlines fetch error:', err)
    error.value = err.response?.data?.error || err.message || 'An unexpected error occurred'
  } finally {
    if (reset) {
      isLoading.value = false
    } else {
      isLoadingMore.value = false
    }
  }
}

const loadMoreHeadlines = async () => {
  currentLimit.value += 4
  await fetchHeadlines(false)
}

const openSummaryModal = (article) => {
  selectedArticle.value = article
  showSummaryModal.value = true
}

const closeSummaryModal = () => {
  showSummaryModal.value = false
  selectedArticle.value = null
}

// Lifecycle
onMounted(async () => {
  await fetchHeadlines()
})
</script>

<style scoped>
/* Hero section gradient animation */
.hero-section {
  background: linear-gradient(135deg, 
    var(--neuron-bg-primary) 0%, 
    var(--neuron-bg-content) 50%, 
    var(--neuron-bg-primary) 100%);
  background-size: 200% 200%;
  animation: gradient-shift 12s ease-in-out infinite;
}

@keyframes gradient-shift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

/* Enhanced content width for better readability */
.content-width {
  max-width: 1200px;
  margin: 0 auto;
}

/* Smooth fade in animation */
.animate-fade-in {
  animation: fade-in 0.6s ease-out;
}

@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Responsive grid adjustments */
@media (max-width: 768px) {
  .hero-section {
    padding: 3rem 1.5rem;
  }
  
  .main-content {
    padding: 2rem 1.5rem;
  }
}
</style>
