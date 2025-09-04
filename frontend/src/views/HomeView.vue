<template>
  <div class="dashboard-view h-full overflow-y-auto">
    
    <!-- Minimalistic Hero Section -->
    <section class="hero-section relative px-6 py-16 bg-gradient-to-br from-neuron-bg-primary via-slate-900/80 to-neuron-bg-primary">
      <div class="content-width">
        <div class="text-center space-y-8">
          
          <!-- Main Heading with lighter font -->
          <h1 class="text-5xl lg:text-6xl xl:text-7xl text-display text-gradient">
            NewsNeuron
          </h1>
          
          <!-- Minimal Subtitle -->
          <p class="text-xl lg:text-2xl text-body-sans text-neuron-text-secondary max-w-3xl mx-auto leading-relaxed font-light">
            AI-curated insights, processed like interconnected neurons.
          </p>
          
        </div>
      </div>
      
      <!-- Subtle animated background -->
      <div class="absolute inset-0 overflow-hidden pointer-events-none">
        <div v-for="i in 12" :key="i" 
             :class="getParticleColor(i)"
             class="absolute w-0.5 h-0.5 rounded-full animate-pulse"
             :style="particleStyle(i)">
        </div>
      </div>
    </section>

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
        <div v-else-if="headlines.length > 0" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <TopHeadlineCard
            v-for="headline in displayedHeadlines"
            :key="headline.id"
            :headline="headline"
            class="animate-fade-in"
          />
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

  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { RefreshCwIcon, AlertCircleIcon } from 'lucide-vue-next'

// Components
import TopHeadlineCard from '@/components/headlines/TopHeadlineCard.vue'
import SynapseLoader from '@/components/ui/SynapseLoader.vue'

// API
import { headlinesAPI } from '@/services/api.js'

// Icons (using a news icon placeholder - you can replace with actual icon)
const NewsIcon = RefreshCwIcon

// State
const isLoading = ref(true)
const error = ref('')
const headlines = ref([])
const selectedCategory = ref('general')

// Computed - Show all headlines (no limit for this minimalistic approach)
const displayedHeadlines = computed(() => {
  return headlines.value
})

// Particle animation with color variety
const particleStyle = () => {
  return {
    left: Math.random() * 100 + '%',
    top: Math.random() * 100 + '%',
    animationDelay: Math.random() * 3 + 's',
    animationDuration: (Math.random() * 3 + 2) + 's'
  }
}

const getParticleColor = (index) => {
  const colors = [
    'bg-neuron-glow/10',
    'bg-accent-emerald/8',
    'bg-accent-violet/8',
    'bg-accent-amber/8',
    'bg-accent-rose/8',
    'bg-accent-cyan/8'
  ]
  return colors[index % colors.length]
}

// Methods
const fetchHeadlines = async () => {
  try {
    isLoading.value = true
    error.value = ''
    
    const response = await headlinesAPI.getTopHeadlines({
      category: selectedCategory.value,
      lang: 'en',
      country: 'us',
      max_articles: 12
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
    isLoading.value = false
  }
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
