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
              <input
                v-model="searchQuery"
                type="text"
                placeholder="Enter entity name (e.g., OpenAI, Tesla, Climate Change)"
                class="input-neuron"
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
              {{ isLoading ? 'Generating...' : 'Generate Timeline' }}
            </button>
          </div>
        </div>

        <!-- Advanced Filters (Removed for cleaner interface) -->
      </div>

      <!-- Timeline Visualization -->
      <div class="max-w-full">
        <!-- Main Timeline -->
        <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <div class="flex items-center justify-between mb-6">
              <div>
                <h2 class="text-xl font-semibold text-text-primary">
                  {{ timelineTitle }}
                </h2>
                <div v-if="hasTimelineData && timelineStatistics?.story_threads" class="flex items-center space-x-4 mt-2 text-sm text-text-secondary">
                  <span class="flex items-center">
                    <span class="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                    {{ timelineStatistics.story_threads }} Story Thread{{ timelineStatistics.story_threads !== 1 ? 's' : '' }}
                  </span>
                  <span v-if="timelineStatistics.standalone_articles > 0" class="flex items-center">
                    <span class="w-2 h-2 bg-gray-500 rounded-full mr-2"></span>
                    {{ timelineStatistics.standalone_articles }} Standalone Article{{ timelineStatistics.standalone_articles !== 1 ? 's' : '' }}
                  </span>
                </div>
              </div>
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
                class="overflow-x-auto"
                :style="{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }"
              >
                <!-- Story Progression Legend -->
                <div v-if="hasTimelineData && timelineStatistics?.story_threads > 0" class="mb-6 p-4 bg-dark-background/30 rounded-lg">
                  <h4 class="text-sm font-medium text-text-primary mb-2">Story Progression Guide</h4>
                  <div class="flex flex-wrap gap-4 text-xs text-text-secondary">
                    <div class="flex items-center">
                      <div class="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
                      <span>Connected story threads</span>
                    </div>
                    <div class="flex items-center">
                      <div class="w-3 h-3 bg-gray-500 rounded-full mr-2"></div>
                      <span>Individual reports</span>
                    </div>
                    <div class="flex items-center">
                      <div class="w-0.5 h-4 bg-purple-500/60 mr-2"></div>
                      <span>Story continuation</span>
                    </div>
                  </div>
                </div>
                
                <!-- Timeline Events Display -->
                <div class="space-y-6">
                  <!-- Story Cards Display -->
                  <div
                    v-for="story in storyCards"
                    :key="story.id"
                    class="story-card bg-dark-background/50 rounded-lg border border-dark-border hover:border-neuron-glow/30 transition-all duration-200"
                  >
                    <!-- Story Card Header -->
                    <div 
                      @click="toggleStoryCard(story.id)"
                      class="p-4 cursor-pointer flex items-center justify-between"
                    >
                      <div class="flex-1">
                        <div class="flex items-center space-x-3 mb-2">
                          <div class="w-3 h-3 rounded-full border-2 border-white"
                               :style="{ backgroundColor: story.color }">
                          </div>
                          <h3 class="text-text-primary font-semibold text-lg">
                            {{ story.title }}
                          </h3>
                          <span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400">
                            {{ story.articleCount }} {{ story.articleCount === 1 ? 'article' : 'articles' }}
                          </span>
                        </div>
                        
                        <p class="text-text-secondary text-sm mb-3">
                          {{ story.summary }}
                        </p>
                        
                        <div class="flex items-center space-x-4 text-xs text-text-muted">
                          <span>{{ formatDate(story.earliestDate) }} → {{ formatDate(story.latestDate) }}</span>
                          <span>Avg. Relevance: {{ Math.round(story.avgRelevance * 100) }}%</span>
                          <span>Sources: {{ story.sources.join(', ') }}</span>
                        </div>
                      </div>
                      
                      <div class="ml-4 text-text-muted">
                        <svg 
                          :class="{ 'rotate-180': isStoryCardExpanded(story.id) }"
                          class="w-5 h-5 transform transition-transform duration-200" 
                          fill="none" 
                          stroke="currentColor" 
                          viewBox="0 0 24 24"
                        >
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                      </div>
                    </div>
                    
                    <!-- Expanded Articles -->
                    <div v-if="isStoryCardExpanded(story.id)" class="border-t border-dark-border">
                      <div class="p-4 space-y-4">
                        <div
                          v-for="(event, articleIndex) in story.articles"
                          :key="event.id"
                          class="timeline-item relative pl-8 pb-4"
                          :class="{ 'timeline-item-last': articleIndex === story.articles.length - 1 }"
                        >
                          <!-- Timeline Dot -->
                          <div class="absolute left-0 top-2 w-2 h-2 rounded-full border border-white"
                               :style="{ backgroundColor: story.color }">
                          </div>
                          
                          <!-- Timeline Line -->
                          <div v-if="articleIndex < story.articles.length - 1" 
                               class="absolute left-1 top-4 w-0.5 h-full bg-dark-border">
                          </div>
                          
                          <!-- Event Content -->
                          <div class="bg-dark-background/30 rounded-lg p-3">
                            <div class="flex items-start justify-between mb-2">
                              <h4 class="text-text-primary font-medium text-sm leading-tight">
                                {{ event.title }}
                              </h4>
                              <span class="text-text-muted text-xs ml-4 flex-shrink-0">
                                {{ formatDate(event.date) }}
                              </span>
                            </div>
                            
                            <p class="text-text-secondary text-sm mb-3 leading-relaxed">
                              {{ event.description }}
                            </p>
                            
                            <div class="flex items-center justify-between">
                              <div class="flex items-center space-x-3">
                                <!-- Source Credibility Badge -->
                                <span class="inline-flex items-center px-2 py-1 rounded-full text-xs"
                                      :class="{
                                        'bg-emerald-500/20 text-emerald-400': getCredibilityTier(event) === 'tier1',
                                        'bg-blue-500/20 text-blue-400': getCredibilityTier(event) === 'tier2',
                                        'bg-yellow-500/20 text-yellow-400': getCredibilityTier(event) === 'tier3',
                                        'bg-gray-500/20 text-gray-400': getCredibilityTier(event) === 'unrated'
                                      }">
                                  {{ getCredibilityLabel(event) }}
                                </span>
                                
                                <!-- Relevance Score -->
                                <span class="text-text-muted text-xs">
                                  Relevance: {{ Math.round((event.relevance_score || event.relevance || 0) * 100) }}%
                                </span>
                                
                                <!-- Source Name -->
                                <span v-if="event.metadata?.source" class="text-text-muted text-xs font-medium">
                                  {{ event.metadata.source }}
                                </span>
                              </div>
                              
                              <a v-if="event.source_url || event.sourceUrl" 
                                 :href="event.source_url || event.sourceUrl" 
                                 target="_blank"
                                 class="text-neuron-glow hover:text-neuron-glow/80 text-xs"
                              >
                                Read more →
                              </a>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Empty State within Timeline -->
                  <div v-if="timelineData.length === 0" class="text-center py-12">
                    <ClockIcon class="w-12 h-12 mx-auto mb-4 text-text-muted" />
                    <p class="text-text-secondary">No events found for this entity and time range.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

  <!-- Toast Container -->
  <div class="fixed bottom-4 right-4 z-50 space-y-2">
    <ToastNotification
      v-for="toast in toasts"
      :key="toast.id"
      :type="toast.type"
      :title="toast.title"
      :message="toast.message"
      @close="removeToast(toast.id)"
    />
  </div>
</template>
<script setup>
import { ref, computed, nextTick } from 'vue'
import { 
  XMarkIcon as XIcon, 
  SparklesIcon, 
  ClockIcon, 
  ArrowPathIcon as RotateCcwIcon, 
  ArrowDownTrayIcon as DownloadIcon,
  MagnifyingGlassPlusIcon as ZoomInIcon,
  MagnifyingGlassMinusIcon as ZoomOutIcon,
  ChartBarIcon as BarChart3Icon,
  ArrowTrendingUpIcon as TrendingUpIcon,
  ShareIcon as NetworkIcon
} from '@heroicons/vue/24/outline'
import SynapseLoader from '@/components/ui/SynapseLoader.vue'
import ToastNotification from '@/components/ui/ToastNotification.vue'
import { timelineAPI } from '@/services/api.js'

// State
const searchQuery = ref('')
const timeRange = ref('30d')
const isLoading = ref(false)
const loadingStatus = ref('')
const currentView = ref('timeline')
const zoomLevel = ref(1)

// Toast notifications state
const toasts = ref([])
let toastIdCounter = 0

// Timeline data
const timelineData = ref([])
const timelineContainer = ref(null)

// Story cards state
const expandedStoryCards = ref(new Set())

// Search history
const recentSearches = ref([])

// Configuration - keeping view modes for future use
const viewModes = [
  { id: 'timeline', label: 'Timeline', icon: BarChart3Icon },
  { id: 'network', label: 'Network', icon: NetworkIcon },
  { id: 'trend', label: 'Trends', icon: TrendingUpIcon }
]

// Computed
const hasTimelineData = computed(() => timelineData.value.length > 0)

// Group timeline events into story cards
const storyCards = computed(() => {
  if (!timelineData.value || timelineData.value.length === 0) return []
  
  // Group events by story cluster
  const clusters = {}
  
  timelineData.value.forEach(event => {
    const clusterId = event.metadata?.story_cluster ?? -1
    
    if (!clusters[clusterId]) {
      clusters[clusterId] = {
        clusterId,
        articles: [],
        earliestDate: event.date,
        latestDate: event.date,
        totalRelevance: 0,
        sources: new Set()
      }
    }
    
    clusters[clusterId].articles.push(event)
    clusters[clusterId].totalRelevance += event.relevance_score || 0
    clusters[clusterId].sources.add(event.metadata?.source || 'Unknown')
    
    // Update date range
    if (new Date(event.date) < new Date(clusters[clusterId].earliestDate)) {
      clusters[clusterId].earliestDate = event.date
    }
    if (new Date(event.date) > new Date(clusters[clusterId].latestDate)) {
      clusters[clusterId].latestDate = event.date
    }
  })
  
  // Convert to array and create story card data
  return Object.values(clusters).map(cluster => {
    const articles = cluster.articles.sort((a, b) => new Date(a.date) - new Date(b.date))
    const avgRelevance = cluster.totalRelevance / articles.length
    
    return {
      id: `story-${cluster.clusterId}`,
      clusterId: cluster.clusterId,
      title: cluster.clusterId >= 0 ? 
        `Story Thread ${cluster.clusterId + 1}` : 
        'Individual Reports',
      summary: articles[0]?.title || 'No title',
      articleCount: articles.length,
      articles: articles,
      earliestDate: cluster.earliestDate,
      latestDate: cluster.latestDate,
      avgRelevance: avgRelevance,
      sources: Array.from(cluster.sources),
      isExpanded: false,
      color: getStoryThreadColor(cluster.clusterId)
    }
  }).sort((a, b) => new Date(a.earliestDate) - new Date(b.earliestDate))
})

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
  loadingStatus.value = 'Checking existing data...'

  try {
    // Prepare timeline request
    const requestData = {
      entity_name: searchQuery.value.trim(),
      time_range: timeRange.value,
      limit: 100
    }

    loadingStatus.value = 'Searching for related articles...'
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // Call the real timeline API
    const response = await timelineAPI.generateTimeline(requestData)
    
    if (response.data.success) {
      // Check if fresh data was fetched
      if (response.data.statistics?.fresh_data_fetched) {
        loadingStatus.value = 'Fresh news fetched! Processing timeline...'
      } else {
        loadingStatus.value = 'Processing existing data...'
      }
      
      await new Promise(resolve => setTimeout(resolve, 300))
      loadingStatus.value = 'Building timeline connections...'
      
      // Transform API response to frontend format
      const timelineEvents = response.data.events.map(event => ({
        id: event.id,
        date: event.date,
        title: event.title,
        description: event.description,
        type: event.event_type,
        sentiment: event.sentiment,
        relevance: event.relevance_score,
        entities: event.entities,
        metadata: event.metadata,
        sourceUrl: event.source_url
      }))

      timelineData.value = timelineEvents
      
      // Update related entities from API response
      if (response.data.related_entities && response.data.related_entities.length > 0) {
        relatedEntities.value = response.data.related_entities.map((entity, index) => ({
          id: index + 1,
          name: entity.name,
          mentions: entity.mentions,
          color: timelineCategories.value[index % timelineCategories.value.length].color
        }))
      }

      // Update recent activity with timeline events
      if (timelineEvents.length > 0) {
        recentActivity.value = timelineEvents.slice(0, 3).map(event => ({
          id: event.id,
          title: event.title.substring(0, 50) + '...',
          description: `${event.sentiment} sentiment • ${Math.round(event.relevance * 100)}% relevance`,
          time: formatRelativeTime(event.date)
        }))
      }

      loadingStatus.value = 'Rendering visualization...'
      await nextTick()
      
      // Show success message with context
      if (response.data.statistics?.fresh_data_fetched) {
        showSuccessMessage(`Timeline generated! Fetched fresh news for "${searchQuery.value}"`)
      } else {
        showSuccessMessage(`Timeline generated for "${searchQuery.value}" using ${response.data.statistics?.articles_analyzed || 0} articles`)
      }
      
      // Add to recent searches
      const searchTerm = searchQuery.value.trim()
      if (!recentSearches.value.includes(searchTerm)) {
        recentSearches.value.unshift(searchTerm)
        recentSearches.value = recentSearches.value.slice(0, 5) // Keep only last 5 searches
      }
      
    } else {
      console.error('Timeline generation failed:', response.data.error)
      timelineData.value = []
      // Show user-friendly error message
      showErrorMessage(`Failed to generate timeline: ${response.data.error || 'Unknown error'}`)
    }

  } catch (error) {
    console.error('Error generating timeline:', error)
    timelineData.value = []
    
    // Handle different types of errors
    let errorMessage = 'Failed to generate timeline. '
    if (error.code === 'NETWORK_ERROR' || error.message.includes('Network Error')) {
      errorMessage += 'Please check your internet connection and try again.'
    } else if (error.response?.status === 500) {
      errorMessage += 'Server error. Please try again later.'
    } else if (error.response?.status === 404) {
      errorMessage += 'Timeline service not available.'
    } else {
      errorMessage += 'Please try again.'
    }
    
    showErrorMessage(errorMessage)
  } finally {
    isLoading.value = false
    loadingStatus.value = ''
  }
}

const exportTimeline = () => {
  if (!hasTimelineData.value) return
  
  try {
    // Prepare export data
    const exportData = {
      entity: searchQuery.value,
      timeRange: timeRange.value,
      generatedAt: new Date().toISOString(),
      totalEvents: timelineData.value.length,
      events: timelineData.value.map(event => ({
        date: event.date,
        title: event.title,
        description: event.description,
        sentiment: event.sentiment,
        relevance: Math.round(event.relevance * 100),
        source: event.metadata?.source || 'Unknown',
        url: event.sourceUrl
      })),
      statistics: stats.value
    }
    
    // Create and download JSON file
    const dataStr = JSON.stringify(exportData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    
    const link = document.createElement('a')
    link.href = URL.createObjectURL(dataBlob)
    link.download = `timeline-${searchQuery.value}-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(link.href)
    
    console.log('Timeline exported successfully')
  } catch (error) {
    console.error('Export failed:', error)
    showErrorMessage('Failed to export timeline. Please try again.')
  }
}

const resetTimeline = () => {
  timelineData.value = []
  searchQuery.value = ''
}

// Story card management functions
const toggleStoryCard = (storyId) => {
  const expanded = new Set(expandedStoryCards.value)
  if (expanded.has(storyId)) {
    expanded.delete(storyId)
  } else {
    expanded.add(storyId)
  }
  expandedStoryCards.value = expanded
}

const isStoryCardExpanded = (storyId) => {
  return expandedStoryCards.value.has(storyId)
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

const formatDate = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const formatRelativeTime = (dateString) => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now - date
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffHours / 24)
  
  if (diffHours < 1) return 'Just now'
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return formatDate(dateString)
}

// Source credibility helper functions
const getCredibilityTier = (event) => {
  const credibility = event.metadata?.source_credibility
  if (!credibility) return 'unrated'
  return credibility.tier || 'unrated'
}

const getCredibilityLabel = (event) => {
  const tier = getCredibilityTier(event)
  const credibility = event.metadata?.source_credibility
  
  if (!credibility) return 'Unrated'
  
  switch (tier) {
    case 'tier1':
      return 'Highly Credible'
    case 'tier2':
      return 'Credible'
    case 'tier3':
      return 'Specialized'
    default:
      return 'Unrated'
  }
}

// Story thread visualization functions
const getStoryThreadColor = (clusterId) => {
  if (clusterId === undefined || clusterId < 0) return 'gray'
  
  const colors = ['purple', 'cyan', 'pink', 'orange', 'green', 'blue', 'red', 'yellow']
  return colors[clusterId % colors.length]
}

const showErrorMessage = (message) => {
  const toast = {
    id: ++toastIdCounter,
    type: 'error',
    title: 'Error',
    message: message,
    duration: 5000
  }
  toasts.value.push(toast)
  
  // Auto-remove after duration
  setTimeout(() => {
    const index = toasts.value.findIndex(t => t.id === toast.id)
    if (index > -1) {
      toasts.value.splice(index, 1)
    }
  }, toast.duration)
}

const showSuccessMessage = (message) => {
  const toast = {
    id: ++toastIdCounter,
    type: 'success',
    title: 'Success',
    message: message,
    duration: 4000
  }
  toasts.value.push(toast)
  
  // Auto-remove after duration
  setTimeout(() => {
    const index = toasts.value.findIndex(t => t.id === toast.id)
    if (index > -1) {
      toasts.value.splice(index, 1)
    }
  }, toast.duration)
}

const removeToast = (toastId) => {
  const index = toasts.value.findIndex(t => t.id === toastId)
  if (index > -1) {
    toasts.value.splice(index, 1)
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

.timeline-item {
  position: relative;
}

.timeline-item::before {
  content: '';
  position: absolute;
  left: 6px;
  top: 0;
  bottom: 0;
  width: 2px;
  background: linear-gradient(to bottom, #374151, transparent);
}

.timeline-item-last::before {
  background: linear-gradient(to bottom, #374151, transparent);
  height: 24px;
}
</style>
