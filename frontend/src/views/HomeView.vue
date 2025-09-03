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

    <!-- Main Content - Just One Row of Cards -->
    <section class="main-content px-6 py-12">
      <div class="content-width">
        
        <!-- Loading State -->
        <div v-if="isLoading" class="flex justify-center items-center py-20">
          <SynapseLoader size="lg" show-text loading-text="Loading insights..." />
        </div>
        
        <!-- Single Row of Insights - No Filters, No Feed Controls -->
        <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          
          <FlashcardPreview
            v-for="flashcard in displayedFlashcards"
            :key="flashcard.id"
            :flashcard="flashcard"
            class="animate-fade-in"
            @click="openFlashcard(flashcard)"
            @bookmark="toggleBookmark(flashcard)"
            @share="shareFlashcard(flashcard)"
          />
          
        </div>
        
      </div>
    </section>

  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

// Components
import FlashcardPreview from '@/components/flashcards/FlashcardPreview.vue'
import SynapseLoader from '@/components/ui/SynapseLoader.vue'

// State
const isLoading = ref(true)

// Mock data for minimalistic design
const mockFlashcards = ref([
  {
    id: 1,
    title: "AI Breakthrough in Neural Networks",
    summary: "Researchers develop new architecture for more efficient neural processing, promising significant advances in machine learning capabilities.",
    category: "Technology",
    created_at: new Date().toISOString(),
    is_bookmarked: false,
    source: "Nature AI",
    reading_time: "3 min read"
  },
  {
    id: 2,
    title: "Climate Change Policy Updates",
    summary: "New international agreements on carbon reduction targets set ambitious goals for the next decade.",
    category: "Environment",
    created_at: new Date().toISOString(),
    is_bookmarked: false,
    source: "Climate Journal",
    reading_time: "4 min read"
  },
  {
    id: 3,
    title: "Quantum Computing Milestone",
    summary: "Scientists achieve new quantum supremacy benchmark, opening doors to revolutionary computing applications.",
    category: "Science",
    created_at: new Date().toISOString(),
    is_bookmarked: false,
    source: "Quantum Tech",
    reading_time: "5 min read"
  },
  {
    id: 4,
    title: "Global Economic Trends",
    summary: "Analysis of emerging market patterns and predictions for sustainable economic growth in developing regions.",
    category: "Economics",
    created_at: new Date().toISOString(),
    is_bookmarked: false,
    source: "Economic Review",
    reading_time: "6 min read"
  }
])

// Computed - Show only one row (4 cards max for minimalistic design)
const displayedFlashcards = computed(() => {
  return mockFlashcards.value.slice(0, 4)
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
const openFlashcard = (flashcard) => {
  console.log('Opening flashcard:', flashcard)
}

const toggleBookmark = (flashcard) => {
  flashcard.is_bookmarked = !flashcard.is_bookmarked
}

const shareFlashcard = (flashcard) => {
  console.log('Sharing flashcard:', flashcard)
}

// Lifecycle
onMounted(async () => {
  // Simulate loading
  setTimeout(() => {
    isLoading.value = false
  }, 800)
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
