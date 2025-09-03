<template>
  <div class="min-h-screen bg-dark-background">
    <!-- Header -->
    <div class="bg-dark-card border-b border-dark-border">
      <div class="max-w-7xl mx-auto px-6 py-8">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold text-text-primary mb-2">Timeline Analysis</h1>
            <p class="text-text-secondary">
              Visualize story evolution and entity connections over time
            </p>
          </div>
          <div class="flex items-center space-x-4">
            <button
              @click="resetTimeline"
              class="btn-secondary"
              :disabled="!hasTimelineData"
            >
              <RotateCcwIcon class="w-4 h-4 mr-2" />
              Reset
            </button>
            <button
              @click="exportTimeline"
              class="btn-neuron"
              :disabled="!hasTimelineData"
            >
              <DownloadIcon class="w-4 h-4 mr-2" />
              Export
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="max-w-7xl mx-auto px-6 py-8">
      <!-- Search & Filters -->
      <div class="bg-dark-card border border-dark-border rounded-2xl p-6 mb-8">
        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <!-- Entity Search -->
          <div class="lg:col-span-2">
            <label class="block text-sm font-medium text-text-primary mb-3">
              Entity or Topic
            </label>
            <div class="relative">
              <SearchIcon class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
              <input
                v-model="searchQuery"
                type="text"
                placeholder="Enter entity name (e.g., OpenAI, Tesla, Climate Change)"
                class="input-neuron pl-12"
                @keydown.enter="generateTimeline"
              />
              <button
                v-if="searchQuery"
                @click="searchQuery = ''"
                class="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary transition-colors"
              >
                <XIcon class="w-4 h-4" />
              </button>
            </div>
          </div>

          <!-- Date Range -->
          <div>
            <label class="block text-sm font-medium text-text-primary mb-3">
              Time Range
            </label>
            <select v-model="timeRange" class="input-neuron">
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 3 months</option>
              <option value="1y">Last year</option>
              <option value="custom">Custom range</option>
            </select>
          </div>

          <!-- Generate Button -->
          <div class="flex items-end">
            <button
              @click="generateTimeline"
              :disabled="!searchQuery.trim() || isLoading"
              class="btn-neuron w-full"
            >
              <SparklesIcon v-if="!isLoading" class="w-4 h-4 mr-2" />
              <SynapseLoader v-else class="w-4 h-4 mr-2" />
              {{ isLoading ? 'Analyzing...' : 'Generate Timeline' }}
            </button>
          </div>
        </div>

        <!-- Advanced Filters -->
        <div v-if="showAdvancedFilters" class="mt-6 pt-6 border-t border-dark-border">
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">
                Event Types
              </label>
              <div class="space-y-2">
                <label v-for="type in eventTypes" :key="type.id" class="flex items-center">
                  <input
                    v-model="selectedEventTypes"
                    :value="type.id"
                    type="checkbox"
                    class="checkbox-neuron mr-3"
                  />
                  <span class="text-text-secondary">{{ type.label }}</span>
                </label>
              </div>
            </div>

            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">
                Sentiment Filter
              </label>
              <select v-model="sentimentFilter" class="input-neuron">
                <option value="all">All sentiments</option>
                <option value="positive">Positive</option>
                <option value="neutral">Neutral</option>
                <option value="negative">Negative</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">
                Min Relevance Score
              </label>
              <div class="flex items-center space-x-3">
                <input
                  v-model="minRelevance"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  class="flex-1"
                />
                <span class="text-text-secondary text-sm w-12">{{ minRelevance }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="mt-4">
          <button
            @click="showAdvancedFilters = !showAdvancedFilters"
            class="text-neuron-glow hover:text-neuron-glow/80 text-sm font-medium transition-colors"
          >
            {{ showAdvancedFilters ? 'Hide' : 'Show' }} Advanced Filters
            <ChevronDownIcon 
              :class="['w-4 h-4 ml-1 inline transition-transform', { 'rotate-180': showAdvancedFilters }]" 
            />
          </button>
        </div>
      </div>

      <!-- Timeline Visualization -->
      <div class="grid grid-cols-1 xl:grid-cols-4 gap-8">
        <!-- Main Timeline -->
        <div class="xl:col-span-3">
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <div class="flex items-center justify-between mb-6">
              <h2 class="text-xl font-semibold text-text-primary">
                {{ timelineTitle }}
              </h2>
              <div v-if="hasTimelineData" class="flex items-center space-x-4">
                <!-- View Toggle -->
                <div class="flex bg-dark-background border border-dark-border rounded-lg p-1">
                  <button
                    v-for="view in viewModes"
                    :key="view.id"
                    @click="currentView = view.id"
                    :class="[
                      'px-3 py-1 rounded text-sm font-medium transition-all',
                      currentView === view.id
                        ? 'bg-neuron-glow text-white'
                        : 'text-text-secondary hover:text-text-primary'
                    ]"
                  >
                    <component :is="view.icon" class="w-4 h-4 mr-1 inline" />
                    {{ view.label }}
                  </button>
                </div>

                <!-- Zoom Controls -->
                <div class="flex items-center space-x-1">
                  <button
                    @click="zoomOut"
                    :disabled="zoomLevel <= 1"
                    class="p-2 rounded-lg hover:bg-dark-background transition-colors disabled:opacity-50"
                  >
                    <ZoomOutIcon class="w-4 h-4 text-text-secondary" />
                  </button>
                  <span class="text-text-muted text-sm px-2">{{ Math.round(zoomLevel * 100) }}%</span>
                  <button
                    @click="zoomIn"
                    :disabled="zoomLevel >= 3"
                    class="p-2 rounded-lg hover:bg-dark-background transition-colors disabled:opacity-50"
                  >
                    <ZoomInIcon class="w-4 h-4 text-text-secondary" />
                  </button>
                </div>
              </div>
            </div>

            <!-- Timeline Container -->
            <div class="relative">
              <div
                v-if="isLoading"
                class="flex items-center justify-center h-96 border-2 border-dashed border-dark-border rounded-xl"
              >
                <div class="text-center">
                  <SynapseLoader class="w-12 h-12 mx-auto mb-4" />
                  <p class="text-text-secondary">Analyzing timeline data...</p>
                  <p class="text-text-muted text-sm mt-1">{{ loadingStatus }}</p>
                </div>
              </div>

              <div
                v-else-if="!hasTimelineData"
                class="flex items-center justify-center h-96 border-2 border-dashed border-dark-border rounded-xl"
              >
                <div class="text-center">
                  <ClockIcon class="w-16 h-16 mx-auto mb-4 text-text-muted" />
                  <h3 class="text-lg font-medium text-text-secondary mb-2">No Timeline Data</h3>
                  <p class="text-text-muted max-w-md">
                    Enter an entity name above and click "Generate Timeline" to visualize story evolution over time.
                  </p>
                </div>
              </div>

              <div
                v-else
                ref="timelineContainer"
                class="timeline-visualization min-h-96 overflow-hidden rounded-xl border border-dark-border"
                :style="{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }"
              >
                <!-- D3.js timeline will be rendered here -->
                <svg
                  ref="timelineSvg"
                  class="w-full h-full"
                  @wheel.prevent="handleWheel"
                ></svg>
              </div>
            </div>

            <!-- Timeline Legend -->
            <div v-if="hasTimelineData" class="mt-6 p-4 bg-dark-background/50 rounded-xl">
              <h4 class="text-sm font-medium text-text-primary mb-3">Legend</h4>
              <div class="flex flex-wrap gap-4">
                <div v-for="category in timelineCategories" :key="category.id" class="flex items-center">
                  <div
                    class="w-3 h-3 rounded-full mr-2"
                    :style="{ backgroundColor: category.color }"
                  ></div>
                  <span class="text-text-secondary text-sm">{{ category.label }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Sidebar -->
        <div class="space-y-6">
          <!-- Timeline Stats -->
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Timeline Stats</h3>
            <div class="space-y-4">
              <div>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-text-secondary text-sm">Total Events</span>
                  <span class="text-text-primary font-semibold">{{ stats.totalEvents }}</span>
                </div>
                <div class="w-full bg-dark-background rounded-full h-2">
                  <div class="bg-neuron-glow h-2 rounded-full" :style="{ width: '100%' }"></div>
                </div>
              </div>

              <div>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-text-secondary text-sm">Articles Analyzed</span>
                  <span class="text-text-primary font-semibold">{{ stats.articlesAnalyzed }}</span>
                </div>
                <div class="w-full bg-dark-background rounded-full h-2">
                  <div class="bg-blue-500 h-2 rounded-full" :style="{ width: `${(stats.articlesAnalyzed / stats.totalEvents) * 100}%` }"></div>
                </div>
              </div>

              <div>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-text-secondary text-sm">Key Entities</span>
                  <span class="text-text-primary font-semibold">{{ stats.keyEntities }}</span>
                </div>
                <div class="w-full bg-dark-background rounded-full h-2">
                  <div class="bg-green-500 h-2 rounded-full" :style="{ width: `${(stats.keyEntities / 20) * 100}%` }"></div>
                </div>
              </div>

              <div>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-text-secondary text-sm">Time Span</span>
                  <span class="text-text-primary font-semibold">{{ stats.timeSpan }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Related Entities -->
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Related Entities</h3>
            <div class="space-y-3">
              <div
                v-for="entity in relatedEntities"
                :key="entity.id"
                class="flex items-center justify-between p-3 bg-dark-background/50 rounded-lg hover:bg-dark-background transition-colors cursor-pointer"
                @click="searchQuery = entity.name; generateTimeline()"
              >
                <div class="flex items-center">
                  <div
                    class="w-2 h-2 rounded-full mr-3"
                    :style="{ backgroundColor: entity.color }"
                  ></div>
                  <span class="text-text-primary text-sm font-medium">{{ entity.name }}</span>
                </div>
                <span class="text-text-muted text-xs">{{ entity.mentions }} mentions</span>
              </div>
            </div>
          </div>

          <!-- Recent Activity -->
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Recent Activity</h3>
            <div class="space-y-3">
              <div
                v-for="activity in recentActivity"
                :key="activity.id"
                class="p-3 bg-dark-background/50 rounded-lg"
              >
                <div class="flex items-start justify-between mb-2">
                  <h4 class="text-text-primary text-sm font-medium">{{ activity.title }}</h4>
                  <span class="text-text-muted text-xs">{{ activity.time }}</span>
                </div>
                <p class="text-text-secondary text-xs">{{ activity.description }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, computed, nextTick } from 'vue'
import { 
  MagnifyingGlassIcon as SearchIcon, 
  XMarkIcon as XIcon, 
  SparklesIcon, 
  ClockIcon, 
  ArrowPathIcon as RotateCcwIcon, 
  ArrowDownTrayIcon as DownloadIcon,
  ChevronDownIcon,
  MagnifyingGlassPlusIcon as ZoomInIcon,
  MagnifyingGlassMinusIcon as ZoomOutIcon,
  ChartBarIcon as BarChart3Icon,
  ArrowTrendingUpIcon as TrendingUpIcon,
  ShareIcon as NetworkIcon
} from '@heroicons/vue/24/outline'
import SynapseLoader from '@/components/ui/SynapseLoader.vue'

// State
const searchQuery = ref('')
const timeRange = ref('30d')
const isLoading = ref(false)
const loadingStatus = ref('')
const showAdvancedFilters = ref(false)
const selectedEventTypes = ref(['news', 'social', 'financial'])
const sentimentFilter = ref('all')
const minRelevance = ref(0.5)
const currentView = ref('timeline')
const zoomLevel = ref(1)

// Timeline data
const timelineData = ref([])
const timelineContainer = ref(null)
const timelineSvg = ref(null)

// Configuration
const eventTypes = [
  { id: 'news', label: 'News Articles' },
  { id: 'social', label: 'Social Media' },
  { id: 'financial', label: 'Financial Reports' },
  { id: 'regulatory', label: 'Regulatory Filings' },
  { id: 'announcements', label: 'Company Announcements' }
]

const viewModes = [
  { id: 'timeline', label: 'Timeline', icon: BarChart3Icon },
  { id: 'network', label: 'Network', icon: NetworkIcon },
  { id: 'trend', label: 'Trends', icon: TrendingUpIcon }
]

// Computed
const hasTimelineData = computed(() => timelineData.value.length > 0)

const timelineTitle = computed(() => {
  if (!searchQuery.value) return 'Timeline Visualization'
  return `Timeline: ${searchQuery.value}`
})

const stats = computed(() => {
  if (!hasTimelineData.value) {
    return {
      totalEvents: 0,
      articlesAnalyzed: 0,
      keyEntities: 0,
      timeSpan: '0 days'
    }
  }

  const events = timelineData.value
  const entities = new Set(events.flatMap(e => e.entities || []))
  const startDate = new Date(Math.min(...events.map(e => new Date(e.date))))
  const endDate = new Date(Math.max(...events.map(e => new Date(e.date))))
  const daysDiff = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24))

  return {
    totalEvents: events.length,
    articlesAnalyzed: events.filter(e => e.type === 'article').length,
    keyEntities: entities.size,
    timeSpan: `${daysDiff} days`
  }
})

const timelineCategories = computed(() => [
  { id: 'news', label: 'News', color: '#00A7E1' },
  { id: 'social', label: 'Social', color: '#10B981' },
  { id: 'financial', label: 'Financial', color: '#F59E0B' },
  { id: 'regulatory', label: 'Regulatory', color: '#EF4444' },
  { id: 'announcement', label: 'Announcements', color: '#8B5CF6' }
])

const relatedEntities = ref([
  { id: 1, name: 'Apple Inc.', mentions: 47, color: '#00A7E1' },
  { id: 2, name: 'Tim Cook', mentions: 23, color: '#10B981' },
  { id: 3, name: 'iPhone', mentions: 89, color: '#F59E0B' },
  { id: 4, name: 'iOS', mentions: 34, color: '#EF4444' }
])

const recentActivity = ref([
  {
    id: 1,
    title: 'Major announcement detected',
    description: 'New product launch mentioned in 12 articles',
    time: '2h ago'
  },
  {
    id: 2,
    title: 'Sentiment shift identified',
    description: 'Positive sentiment increased by 15%',
    time: '4h ago'
  },
  {
    id: 3,
    title: 'Entity connection discovered',
    description: 'New relationship between entities found',
    time: '6h ago'
  }
])

// Methods
const generateTimeline = async () => {
  if (!searchQuery.value.trim()) return

  isLoading.value = true
  loadingStatus.value = 'Searching for related articles...'

  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    loadingStatus.value = 'Analyzing entity mentions...'
    
    await new Promise(resolve => setTimeout(resolve, 1000))
    loadingStatus.value = 'Building timeline connections...'
    
    await new Promise(resolve => setTimeout(resolve, 800))
    loadingStatus.value = 'Rendering visualization...'

    // Generate mock timeline data
    const mockData = generateMockTimelineData()
    timelineData.value = mockData

    await nextTick()
    // Note: D3.js integration would go here in a real implementation

  } catch (error) {
    console.error('Error generating timeline:', error)
  } finally {
    isLoading.value = false
    loadingStatus.value = ''
  }
}

const generateMockTimelineData = () => {
  const now = new Date()
  const data = []
  const days = timeRange.value === '7d' ? 7 : timeRange.value === '30d' ? 30 : 90

  for (let i = 0; i < days; i++) {
    const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000)
    const events = Math.floor(Math.random() * 5)
    
    for (let j = 0; j < events; j++) {
      data.push({
        id: `${i}-${j}`,
        date: date.toISOString(),
        title: `Event related to ${searchQuery.value}`,
        type: eventTypes[Math.floor(Math.random() * eventTypes.length)].id,
        sentiment: Math.random() > 0.5 ? 'positive' : Math.random() > 0.5 ? 'negative' : 'neutral',
        relevance: Math.random(),
        entities: [searchQuery.value],
        description: `Important development involving ${searchQuery.value} detected in news analysis.`
      })
    }
  }

  return data.sort((a, b) => new Date(a.date) - new Date(b.date))
}

const resetTimeline = () => {
  timelineData.value = []
  searchQuery.value = ''
}

const exportTimeline = () => {
  // Export timeline functionality
  console.log('Exporting timeline...')
}

const zoomIn = () => {
  if (zoomLevel.value < 3) {
    zoomLevel.value += 0.2
  }
}

const zoomOut = () => {
  if (zoomLevel.value > 1) {
    zoomLevel.value -= 0.2
  }
}

const handleWheel = (event) => {
  event.preventDefault()
  if (event.deltaY < 0) {
    zoomIn()
  } else {
    zoomOut()
  }
}
</script>

<style scoped>
.timeline-visualization {
  background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
}

.input-neuron {
  @apply w-full px-4 py-3 bg-dark-background border border-dark-border rounded-xl;
  @apply text-text-primary placeholder-text-secondary;
  @apply focus:ring-2 focus:ring-neuron-glow/30 focus:border-neuron-glow transition-all;
}

.btn-neuron {
  @apply px-6 py-3 bg-neuron-glow text-white font-medium rounded-xl;
  @apply hover:bg-neuron-glow/80 transition-all duration-200;
  @apply disabled:opacity-50 disabled:cursor-not-allowed;
}

.btn-secondary {
  @apply px-6 py-3 bg-dark-background border border-dark-border text-text-primary font-medium rounded-xl;
  @apply hover:bg-dark-border transition-all duration-200;
  @apply disabled:opacity-50 disabled:cursor-not-allowed;
}

.checkbox-neuron {
  @apply w-4 h-4 rounded border-dark-border bg-dark-background;
  @apply focus:ring-2 focus:ring-neuron-glow/30;
  @apply checked:bg-neuron-glow checked:border-neuron-glow;
}
</style>
