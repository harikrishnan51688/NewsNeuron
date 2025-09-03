<template>
  <div class="min-h-screen bg-dark-background">
    <!-- Header -->
    <div class="bg-dark-card border-b border-dark-border">
      <div class="max-w-7xl mx-auto px-6 py-8">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl text-display text-text-primary mb-2">Semantic Search</h1>
            <p class="text-body-sans text-text-secondary">
              AI-powered search through news articles and entity connections
            </p>
          </div>
          <div class="flex items-center space-x-4">
            <button
              @click="clearSearch"
              class="btn-secondary"
              :disabled="!hasSearched"
            >
              <RotateCcwIcon class="w-4 h-4 mr-2" />
              Clear
            </button>
            <button
              @click="exportResults"
              class="btn-neuron"
              :disabled="!hasResults"
            >
              <DownloadIcon class="w-4 h-4 mr-2" />
              Export
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="max-w-7xl mx-auto px-6 py-8">
      <!-- Search Bar -->
      <div class="bg-dark-card border border-dark-border rounded-2xl p-6 mb-8">
        <div class="flex items-center space-x-4 mb-6">
          <div class="relative flex-1">
            <SearchIcon class="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search for articles, entities, topics, or complex queries..."
              class="input-neuron pl-14 pr-12 h-14 text-lg"
              @keydown.enter="performSearch"
              @input="handleSearchInput"
            />
            <button
              v-if="searchQuery"
              @click="searchQuery = ''"
              class="absolute right-4 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary transition-colors"
            >
              <XIcon class="w-5 h-5" />
            </button>
          </div>
          <button
            @click="performSearch"
            :disabled="!searchQuery.trim() || isSearching"
            class="btn-neuron h-14 px-8"
          >
            <SparklesIcon v-if="!isSearching" class="w-5 h-5 mr-3" />
            <SynapseLoader v-else class="w-5 h-5 mr-3" />
            {{ isSearching ? 'Searching...' : 'Search' }}
          </button>
        </div>

        <!-- Search Suggestions -->
        <div v-if="searchSuggestions.length > 0 && !hasSearched" class="flex flex-wrap gap-2">
          <span class="text-text-secondary text-sm mr-3">Suggestions:</span>
          <button
            v-for="suggestion in searchSuggestions"
            :key="suggestion"
            @click="searchQuery = suggestion; performSearch()"
            class="px-3 py-1 bg-dark-background border border-dark-border rounded-lg text-text-secondary hover:text-text-primary hover:border-neuron-glow/50 text-sm transition-all"
          >
            {{ suggestion }}
          </button>
        </div>

        <!-- Advanced Search Toggle -->
        <div class="mt-4">
          <button
            @click="showAdvancedSearch = !showAdvancedSearch"
            class="text-neuron-glow hover:text-neuron-glow/80 text-sm font-medium transition-colors"
          >
            {{ showAdvancedSearch ? 'Hide' : 'Show' }} Advanced Search
            <ChevronDownIcon 
              :class="['w-4 h-4 ml-1 inline transition-transform', { 'rotate-180': showAdvancedSearch }]" 
            />
          </button>
        </div>

        <!-- Advanced Search Filters -->
        <div v-if="showAdvancedSearch" class="mt-6 pt-6 border-t border-dark-border">
          <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">
                Search Type
              </label>
              <select v-model="searchType" class="input-neuron">
                <option value="semantic">Semantic Search</option>
                <option value="keyword">Keyword Search</option>
                <option value="entity">Entity Search</option>
                <option value="topic">Topic Search</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">
                Date Range
              </label>
              <select v-model="dateRange" class="input-neuron">
                <option value="all">All Time</option>
                <option value="today">Today</option>
                <option value="week">Last Week</option>
                <option value="month">Last Month</option>
                <option value="year">Last Year</option>
                <option value="custom">Custom Range</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">
                Source Type
              </label>
              <select v-model="sourceType" class="input-neuron">
                <option value="all">All Sources</option>
                <option value="news">News Articles</option>
                <option value="social">Social Media</option>
                <option value="financial">Financial Reports</option>
                <option value="research">Research Papers</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">
                Relevance Score
              </label>
              <div class="flex items-center space-x-3">
                <input
                  v-model="minRelevanceScore"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  class="flex-1"
                />
                <span class="text-text-secondary text-sm w-12">{{ minRelevanceScore }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Search Results -->
      <div class="grid grid-cols-1 xl:grid-cols-4 gap-8">
        <!-- Main Results -->
        <div class="xl:col-span-3">
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <!-- Results Header -->
            <div class="flex items-center justify-between mb-6">
              <div>
                <h2 class="text-xl font-semibold text-text-primary">
                  {{ resultsTitle }}
                </h2>
                <p v-if="hasResults" class="text-text-secondary text-sm mt-1">
                  Found {{ searchResults.length }} results in {{ searchTime }}ms
                </p>
              </div>
              
              <div v-if="hasResults" class="flex items-center space-x-4">
                <!-- Sort Options -->
                <select v-model="sortBy" class="input-neuron w-48">
                  <option value="relevance">Sort by Relevance</option>
                  <option value="date">Sort by Date</option>
                  <option value="popularity">Sort by Popularity</option>
                  <option value="sentiment">Sort by Sentiment</option>
                </select>

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
              </div>
            </div>

            <!-- Search Results Content -->
            <div class="relative">
              <!-- Loading State -->
              <div
                v-if="isSearching"
                class="flex items-center justify-center h-96 border-2 border-dashed border-dark-border rounded-xl"
              >
                <div class="text-center">
                  <SynapseLoader class="w-12 h-12 mx-auto mb-4" />
                  <p class="text-text-secondary">Searching through articles...</p>
                  <p class="text-text-muted text-sm mt-1">{{ searchingStatus }}</p>
                </div>
              </div>

              <!-- Empty State -->
              <div
                v-else-if="!hasResults && !hasSearched"
                class="flex items-center justify-center h-96 border-2 border-dashed border-dark-border rounded-xl"
              >
                <div class="text-center">
                  <SearchIcon class="w-16 h-16 mx-auto mb-4 text-text-muted" />
                  <h3 class="text-lg font-medium text-text-secondary mb-2">Start Your Search</h3>
                  <p class="text-text-muted max-w-md">
                    Enter keywords, topics, or entities above to find relevant news articles and insights.
                  </p>
                </div>
              </div>

              <!-- No Results -->
              <div
                v-else-if="!hasResults && hasSearched"
                class="flex items-center justify-center h-64 border-2 border-dashed border-dark-border rounded-xl"
              >
                <div class="text-center">
                  <SearchIcon class="w-12 h-12 mx-auto mb-4 text-text-muted" />
                  <h3 class="text-lg font-medium text-text-secondary mb-2">No Results Found</h3>
                  <p class="text-text-muted max-w-md">
                    Try adjusting your search terms or filters to find more results.
                  </p>
                </div>
              </div>

              <!-- Results List -->
              <div v-else class="space-y-6">
                <div
                  v-for="result in sortedResults"
                  :key="result.id"
                  class="border border-dark-border rounded-xl p-6 hover:border-neuron-glow/50 transition-all cursor-pointer group"
                  @click="openArticle(result)"
                >
                  <div class="flex items-start justify-between mb-4">
                    <div class="flex-1">
                      <h3 class="text-lg font-semibold text-text-primary group-hover:text-neuron-glow transition-colors">
                        {{ result.title }}
                      </h3>
                      <p class="text-text-secondary mt-2 line-clamp-2">
                        {{ result.summary }}
                      </p>
                    </div>
                    <div class="ml-4 flex-shrink-0">
                      <div class="flex items-center space-x-2">
                        <div
                          :class="[
                            'px-2 py-1 rounded-lg text-xs font-medium',
                            getRelevanceClass(result.relevance_score)
                          ]"
                        >
                          {{ (result.relevance_score * 100).toFixed(0) }}% match
                        </div>
                        <ExternalLinkIcon class="w-4 h-4 text-text-muted group-hover:text-neuron-glow transition-colors" />
                      </div>
                    </div>
                  </div>

                  <div class="flex items-center justify-between text-sm">
                    <div class="flex items-center space-x-4 text-text-muted">
                      <span class="flex items-center">
                        <CalendarIcon class="w-4 h-4 mr-1" />
                        {{ formatDate(result.published_date) }}
                      </span>
                      <span class="flex items-center">
                        <UserIcon class="w-4 h-4 mr-1" />
                        {{ result.source }}
                      </span>
                      <span class="flex items-center">
                        <TagIcon class="w-4 h-4 mr-1" />
                        {{ result.category }}
                      </span>
                    </div>

                    <div class="flex items-center space-x-3">
                      <!-- Sentiment Indicator -->
                      <div
                        :class="[
                          'w-2 h-2 rounded-full',
                          getSentimentColor(result.sentiment)
                        ]"
                      ></div>
                      <span class="text-text-muted capitalize">{{ result.sentiment }}</span>
                    </div>
                  </div>

                  <!-- Entity Tags -->
                  <div v-if="result.entities && result.entities.length > 0" class="mt-4">
                    <div class="flex flex-wrap gap-2">
                      <span
                        v-for="entity in result.entities.slice(0, 5)"
                        :key="entity"
                        class="px-2 py-1 bg-dark-background border border-dark-border rounded-lg text-xs text-text-secondary hover:border-neuron-glow/50 transition-colors cursor-pointer"
                        @click.stop="searchQuery = entity; performSearch()"
                      >
                        {{ entity }}
                      </span>
                      <span
                        v-if="result.entities.length > 5"
                        class="px-2 py-1 text-xs text-text-muted"
                      >
                        +{{ result.entities.length - 5 }} more
                      </span>
                    </div>
                  </div>
                </div>

                <!-- Load More -->
                <div v-if="hasMoreResults" class="text-center pt-6">
                  <button
                    @click="loadMoreResults"
                    :disabled="isLoadingMore"
                    class="btn-secondary"
                  >
                    <RefreshCwIcon v-if="isLoadingMore" class="w-4 h-4 mr-2 animate-spin" />
                    {{ isLoadingMore ? 'Loading...' : 'Load More Results' }}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Sidebar -->
        <div class="space-y-6">
          <!-- Search Insights -->
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Search Insights</h3>
            <div class="space-y-4">
              <div>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-text-secondary text-sm">Relevance Score</span>
                  <span class="text-text-primary font-semibold">{{ avgRelevance }}%</span>
                </div>
                <div class="w-full bg-dark-background rounded-full h-2">
                  <div class="bg-neuron-glow h-2 rounded-full" :style="{ width: `${avgRelevance}%` }"></div>
                </div>
              </div>

              <div>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-text-secondary text-sm">Coverage</span>
                  <span class="text-text-primary font-semibold">{{ searchCoverage }}%</span>
                </div>
                <div class="w-full bg-dark-background rounded-full h-2">
                  <div class="bg-blue-500 h-2 rounded-full" :style="{ width: `${searchCoverage}%` }"></div>
                </div>
              </div>

              <div class="pt-2 border-t border-dark-border">
                <div class="text-text-secondary text-sm space-y-1">
                  <div class="flex justify-between">
                    <span>Articles found:</span>
                    <span>{{ searchResults.length }}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Entities identified:</span>
                    <span>{{ uniqueEntities }}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Time range:</span>
                    <span>{{ searchTimeRange }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Related Entities -->
          <div v-if="relatedEntities.length > 0" class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Related Entities</h3>
            <div class="space-y-3">
              <div
                v-for="entity in relatedEntities"
                :key="entity.name"
                class="flex items-center justify-between p-3 bg-dark-background/50 rounded-lg hover:bg-dark-background transition-colors cursor-pointer"
                @click="searchQuery = entity.name; performSearch()"
              >
                <span class="text-text-primary text-sm font-medium">{{ entity.name }}</span>
                <span class="text-text-muted text-xs">{{ entity.mentions }} mentions</span>
              </div>
            </div>
          </div>

          <!-- Search History -->
          <div v-if="searchHistory.length > 0" class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Recent Searches</h3>
            <div class="space-y-2">
              <div
                v-for="search in searchHistory"
                :key="search.query"
                class="flex items-center justify-between p-2 rounded-lg hover:bg-dark-background/50 transition-colors cursor-pointer"
                @click="searchQuery = search.query; performSearch()"
              >
                <span class="text-text-secondary text-sm">{{ search.query }}</span>
                <span class="text-text-muted text-xs">{{ search.results }} results</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import {
  MagnifyingGlassIcon as SearchIcon,
  SparklesIcon,
  ArrowPathIcon as RotateCcwIcon,
  ArrowDownTrayIcon as DownloadIcon,
  XMarkIcon as XIcon,
  ChevronDownIcon,
  ArrowTopRightOnSquareIcon as ExternalLinkIcon,
  CalendarIcon,
  UserIcon,
  TagIcon,
  ArrowPathIcon as RefreshCwIcon,
  ListBulletIcon,
  Squares2X2Icon,
  ChartBarIcon
} from '@heroicons/vue/24/outline'
import SynapseLoader from '@/components/ui/SynapseLoader.vue'

export default {
  name: 'SearchView',
  components: {
    SearchIcon,
    SparklesIcon,
    RotateCcwIcon,
    DownloadIcon,
    XIcon,
    ChevronDownIcon,
    ExternalLinkIcon,
    CalendarIcon,
    UserIcon,
    TagIcon,
    RefreshCwIcon,
    ListBulletIcon,
    Squares2X2Icon,
    ChartBarIcon,
    SynapseLoader
  },
  setup() {
    const router = useRouter()

    // Search State
    const searchQuery = ref('')
    const isSearching = ref(false)
    const hasSearched = ref(false)
    const searchResults = ref([])
    const searchTime = ref(0)
    const searchingStatus = ref('')

    // UI State
    const showAdvancedSearch = ref(false)
    const currentView = ref('list')
    const isLoadingMore = ref(false)
    const hasMoreResults = ref(false)

    // Search Options
    const searchType = ref('semantic')
    const dateRange = ref('all')
    const sourceType = ref('all')
    const minRelevanceScore = ref(0.3)
    const sortBy = ref('relevance')

    // View Modes
    const viewModes = [
      { id: 'list', label: 'List', icon: ListBulletIcon },
      { id: 'grid', label: 'Grid', icon: Squares2X2Icon },
      { id: 'analytics', label: 'Analytics', icon: ChartBarIcon }
    ]

    // Search Suggestions
    const searchSuggestions = ref([
      'Tesla quarterly earnings',
      'AI technology trends',
      'Climate change policy',
      'Cryptocurrency regulations',
      'Healthcare innovation',
      'Space exploration missions'
    ])

    // Search History
    const searchHistory = ref([
      { query: 'AI revolution', results: 342 },
      { query: 'Climate summit 2024', results: 128 },
      { query: 'Tesla stock analysis', results: 89 }
    ])

    // Related Entities
    const relatedEntities = ref([])

    // Computed Properties
    const hasResults = computed(() => searchResults.value.length > 0)
    
    const resultsTitle = computed(() => {
      if (!hasSearched.value) return 'Search Results'
      if (isSearching.value) return 'Searching...'
      if (!hasResults.value) return 'No Results Found'
      return `Search Results for "${searchQuery.value}"`
    })

    const sortedResults = computed(() => {
      const results = [...searchResults.value]
      
      switch (sortBy.value) {
        case 'date':
          return results.sort((a, b) => new Date(b.published_date) - new Date(a.published_date))
        case 'popularity':
          return results.sort((a, b) => b.engagement_score - a.engagement_score)
        case 'sentiment':
          return results.sort((a, b) => {
            const sentimentOrder = { positive: 3, neutral: 2, negative: 1 }
            return sentimentOrder[b.sentiment] - sentimentOrder[a.sentiment]
          })
        default: // relevance
          return results.sort((a, b) => b.relevance_score - a.relevance_score)
      }
    })

    const avgRelevance = computed(() => {
      if (!hasResults.value) return 0
      const avg = searchResults.value.reduce((sum, result) => sum + result.relevance_score, 0) / searchResults.value.length
      return Math.round(avg * 100)
    })

    const searchCoverage = computed(() => {
      if (!hasResults.value) return 0
      return Math.min(100, Math.round((searchResults.value.length / 1000) * 100))
    })

    const uniqueEntities = computed(() => {
      if (!hasResults.value) return 0
      const allEntities = searchResults.value.flatMap(result => result.entities || [])
      return new Set(allEntities).size
    })

    const searchTimeRange = computed(() => {
      if (!hasResults.value) return 'N/A'
      const dates = searchResults.value.map(result => new Date(result.published_date))
      const earliest = new Date(Math.min(...dates))
      const latest = new Date(Math.max(...dates))
      const diffDays = Math.ceil((latest - earliest) / (1000 * 60 * 60 * 24))
      return `${diffDays} days`
    })

    // Methods
    const generateMockResults = (query) => {
      const mockResults = [
        {
          id: 1,
          title: `Breaking: ${query} impacts global markets as investors react`,
          summary: `Recent developments in ${query} have sent shockwaves through financial markets worldwide. Analysts are closely monitoring the situation as trading volumes surge.`,
          published_date: '2024-01-15T10:30:00Z',
          source: 'Reuters',
          category: 'Finance',
          relevance_score: 0.92,
          sentiment: 'negative',
          engagement_score: 1250,
          entities: [query, 'Stock Market', 'Global Economy', 'Investors']
        },
        {
          id: 2,
          title: `Technology firms adapt to ${query} with innovative solutions`,
          summary: `Leading technology companies are developing new approaches to address challenges related to ${query}. The innovations could reshape entire industries.`,
          published_date: '2024-01-14T15:45:00Z',
          source: 'TechCrunch',
          category: 'Technology',
          relevance_score: 0.87,
          sentiment: 'positive',
          engagement_score: 890,
          entities: [query, 'Technology', 'Innovation', 'Startups']
        },
        {
          id: 3,
          title: `${query} policy changes expected to affect millions`,
          summary: `Government officials announce significant policy shifts regarding ${query}. The changes are set to take effect next quarter with widespread implications.`,
          published_date: '2024-01-13T09:15:00Z',
          source: 'Associated Press',
          category: 'Politics',
          relevance_score: 0.81,
          sentiment: 'neutral',
          engagement_score: 2100,
          entities: [query, 'Government', 'Policy', 'Regulation']
        },
        {
          id: 4,
          title: `Research reveals new insights about ${query} effects`,
          summary: `Comprehensive study published in leading journal shows unexpected correlations between ${query} and various economic indicators.`,
          published_date: '2024-01-12T14:20:00Z',
          source: 'Nature',
          category: 'Research',
          relevance_score: 0.76,
          sentiment: 'positive',
          engagement_score: 445,
          entities: [query, 'Research', 'Study', 'Science']
        },
        {
          id: 5,
          title: `Global summit addresses ${query} concerns`,
          summary: `World leaders gather to discuss international cooperation on ${query}-related issues. Key agreements reached on collaborative frameworks.`,
          published_date: '2024-01-11T11:00:00Z',
          source: 'BBC',
          category: 'International',
          relevance_score: 0.73,
          sentiment: 'positive',
          engagement_score: 1650,
          entities: [query, 'International Relations', 'Summit', 'Cooperation']
        }
      ]

      // Add some randomization to relevance scores
      return mockResults.map(result => ({
        ...result,
        relevance_score: Math.max(0.5, result.relevance_score + (Math.random() - 0.5) * 0.2)
      }))
    }

    const updateRelatedEntities = (results) => {
      const entityCounts = {}
      results.forEach(result => {
        (result.entities || []).forEach(entity => {
          entityCounts[entity] = (entityCounts[entity] || 0) + 1
        })
      })

      relatedEntities.value = Object.entries(entityCounts)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 8)
        .map(([name, mentions]) => ({ name, mentions }))
    }

    const performSearch = async () => {
      if (!searchQuery.value.trim() || isSearching.value) return

      isSearching.value = true
      hasSearched.value = true
      searchingStatus.value = 'Analyzing query...'

      try {
        // Simulate search process
        await new Promise(resolve => setTimeout(resolve, 500))
        searchingStatus.value = 'Searching articles...'
        
        await new Promise(resolve => setTimeout(resolve, 800))
        searchingStatus.value = 'Processing results...'
        
        await new Promise(resolve => setTimeout(resolve, 400))

        const startTime = Date.now()
        const results = generateMockResults(searchQuery.value)
        const endTime = Date.now()

        searchResults.value = results.filter(result => 
          result.relevance_score >= minRelevanceScore.value
        )
        searchTime.value = endTime - startTime + Math.random() * 200

        updateRelatedEntities(searchResults.value)

        // Update search history
        const existingIndex = searchHistory.value.findIndex(
          item => item.query === searchQuery.value
        )
        if (existingIndex >= 0) {
          searchHistory.value.splice(existingIndex, 1)
        }
        searchHistory.value.unshift({
          query: searchQuery.value,
          results: searchResults.value.length
        })
        searchHistory.value = searchHistory.value.slice(0, 5)

        hasMoreResults.value = Math.random() > 0.5

      } catch (error) {
        console.error('Search error:', error)
        searchResults.value = []
      } finally {
        isSearching.value = false
        searchingStatus.value = ''
      }
    }

    const handleSearchInput = () => {
      // Could implement search suggestions here
    }

    const clearSearch = () => {
      searchQuery.value = ''
      searchResults.value = []
      hasSearched.value = false
      relatedEntities.value = []
    }

    const loadMoreResults = async () => {
      isLoadingMore.value = true
      
      try {
        await new Promise(resolve => setTimeout(resolve, 1000))
        const moreResults = generateMockResults(searchQuery.value + ' extended')
        searchResults.value.push(...moreResults.slice(0, 3))
        hasMoreResults.value = Math.random() > 0.7
      } finally {
        isLoadingMore.value = false
      }
    }

    const exportResults = () => {
      const data = searchResults.value.map(result => ({
        title: result.title,
        summary: result.summary,
        date: result.published_date,
        source: result.source,
        relevance: result.relevance_score
      }))
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `search-results-${searchQuery.value}-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
    }

    const openArticle = (result) => {
      // In a real app, this would open the article
      console.log('Opening article:', result.title)
    }

    const formatDate = (dateString) => {
      const date = new Date(dateString)
      const now = new Date()
      const diffMs = now - date
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
      
      if (diffDays === 0) return 'Today'
      if (diffDays === 1) return 'Yesterday'
      if (diffDays < 7) return `${diffDays} days ago`
      return date.toLocaleDateString()
    }

    const getRelevanceClass = (score) => {
      if (score >= 0.9) return 'bg-green-500/20 text-green-400 border border-green-500/30'
      if (score >= 0.7) return 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
      if (score >= 0.5) return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
      return 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
    }

    const getSentimentColor = (sentiment) => {
      switch (sentiment) {
        case 'positive': return 'bg-green-500'
        case 'negative': return 'bg-red-500'
        default: return 'bg-gray-500'
      }
    }

    return {
      // State
      searchQuery,
      isSearching,
      hasSearched,
      searchResults,
      searchTime,
      searchingStatus,
      showAdvancedSearch,
      currentView,
      isLoadingMore,
      hasMoreResults,
      searchType,
      dateRange,
      sourceType,
      minRelevanceScore,
      sortBy,
      viewModes,
      searchSuggestions,
      searchHistory,
      relatedEntities,

      // Computed
      hasResults,
      resultsTitle,
      sortedResults,
      avgRelevance,
      searchCoverage,
      uniqueEntities,
      searchTimeRange,

      // Methods
      performSearch,
      handleSearchInput,
      clearSearch,
      loadMoreResults,
      exportResults,
      openArticle,
      formatDate,
      getRelevanceClass,
      getSentimentColor
    }
  }
}
</script>
