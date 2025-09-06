<template>
  <div class="news-flashcard-app min-h-screen bg-neuron-bg-primary">
    
    <!-- Subtle Background Particles -->
    <div class="fixed inset-0 overflow-hidden pointer-events-none">
      <div v-for="i in 8" :key="i" 
           :class="getParticleColor(i)" 
           class="absolute w-0.5 h-0.5 rounded-full animate-pulse"
           :style="particleStyle(i)">
      </div>
    </div>

    <div class="max-w-6xl mx-auto px-6 py-8 relative z-10">
      
      <!-- Header -->
      <div class="text-center mb-8">
        <div class="w-16 h-16 bg-gradient-to-br from-accent-emerald to-accent-cyan rounded-full mx-auto flex items-center justify-center mb-4">
          <Brain class="w-8 h-8 text-white" />
        </div>
        <h1 class="text-3xl font-bold text-neuron-text-primary mb-2">NewsNeuron Flashcards</h1>
        <p class="text-neuron-text-secondary">Stay informed with AI-powered news learning</p>
      </div>

      <!-- News Topics Selection -->
      <div class="bg-neuron-bg-content/50 backdrop-blur-sm border border-neuron-border rounded-2xl p-6 mb-8">
        <h3 class="text-lg font-semibold text-neuron-text-primary mb-4 flex items-center">
          <Newspaper class="w-5 h-5 mr-2 text-accent-emerald" />
          Select News Topic
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          <button v-for="topic in newsTopics" :key="topic"
                  @click="selectedTopic = topic"
                  :class="['p-3 rounded-xl border transition-all text-sm font-medium', 
                         selectedTopic === topic 
                         ? 'bg-gradient-to-r from-accent-emerald to-accent-cyan text-white border-transparent' 
                         : 'bg-neuron-bg-content border-neuron-border text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/50']">
            {{ getTopicEmoji(topic) }} {{ topic }}
          </button>
        </div>
      </div>

      <!-- Stats Dashboard -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div v-for="(stat, key) in stats" :key="key" 
             class="bg-neuron-bg-content border border-neuron-border rounded-xl p-4 hover:border-accent-emerald/30 transition-colors">
          <div class="flex items-center justify-between">
            <div>
              <div class="text-2xl font-bold text-neuron-text-primary">{{ stat.value }}</div>
              <div class="text-sm text-neuron-text-secondary">{{ stat.label }}</div>
            </div>
            <div :class="stat.color" class="w-10 h-10 rounded-lg flex items-center justify-center">
              <div class="text-lg">{{ stat.icon }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Controls -->
      <div class="flex flex-wrap gap-4 mb-8 justify-center">
        <select v-model="selectedCategory" 
                @change="filterCards"
                class="px-4 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-primary focus:outline-none focus:border-accent-emerald/50">
          <option value="all">All Categories</option>
          <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
        </select>
        
        <select v-model="sourceType" 
                class="px-4 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-primary focus:outline-none focus:border-accent-emerald/50">
          <option value="recent_news">Recent News</option>
          <option value="knowledge_base">Stored Articles</option>
          <option value="entities">Knowledge Graph</option>
        </select>
        
        <button @click="shuffleCards" 
                class="px-4 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/50 transition-colors flex items-center gap-2">
          <Shuffle class="w-4 h-4" />
          Shuffle
        </button>
        
        <button @click="generateNewsCards" 
                :disabled="loading"
                class="px-6 py-2 bg-gradient-to-r from-accent-emerald to-accent-cyan text-white rounded-xl hover:shadow-lg transition-all disabled:opacity-50 flex items-center gap-2">
          <Loader2 v-if="loading" class="w-4 h-4 animate-spin" />
          <Newspaper v-else class="w-4 h-4" />
          {{ loading ? 'Generating...' : 'Generate Cards' }}
        </button>
        
        <button @click="startStudySession" 
                class="px-4 py-2 bg-gradient-to-r from-accent-violet to-accent-rose text-white rounded-xl hover:shadow-lg transition-all flex items-center gap-2">
          <BookOpen class="w-4 h-4" />
          Study Session
        </button>
      </div>

      <!-- Study Session Info -->
      <div v-if="studyMode" class="bg-gradient-to-r from-accent-emerald/20 to-accent-cyan/20 border border-accent-emerald/30 rounded-xl p-4 mb-8">
        <div class="flex justify-between items-center">
          <h3 class="text-lg font-semibold text-neuron-text-primary flex items-center gap-2">
            <BookOpen class="w-5 h-5 text-accent-emerald" />
            Study Session Active
          </h3>
          <div class="flex gap-4 text-sm">
            <span class="text-green-400">✅ {{ sessionStats.correct }}</span>
            <span class="text-red-400">❌ {{ sessionStats.incorrect }}</span>
            <span class="text-neuron-text-secondary">Total: {{ sessionStats.total }}</span>
          </div>
        </div>
      </div>

      <!-- Flashcard Container -->
      <div v-if="filteredCards.length > 0" class="mb-8">
        <!-- Progress Bar -->
        <div class="mb-6">
          <div class="flex justify-between text-sm text-neuron-text-secondary mb-2">
            <span>{{ currentIndex + 1 }} / {{ filteredCards.length }}</span>
            <span>{{ Math.round(progressPercentage) }}% Complete</span>
          </div>
          <div class="w-full bg-neuron-bg-content rounded-full h-2 border border-neuron-border">
            <div class="bg-gradient-to-r from-accent-emerald to-accent-cyan rounded-full h-full transition-all duration-300"
                 :style="{ width: progressPercentage + '%' }"></div>
          </div>
        </div>

        <!-- Flashcard -->
        <div class="relative flashcard-container" @click="flipCard">
          <div :class="['flashcard-inner transition-transform duration-700', 
                        { 'flipped': isFlipped }]">
            
            <!-- Front of Card -->
            <div class="flashcard-face front absolute w-full bg-neuron-bg-content border border-neuron-border rounded-2xl p-8 min-h-[400px] flex flex-col cursor-pointer hover:border-accent-emerald/30 transition-colors">
              <div class="flex justify-between items-start mb-6">
                <span class="px-3 py-1 bg-accent-emerald/20 text-accent-emerald rounded-full text-sm font-medium">
                  {{ currentCard.category }}
                </span>
                <div class="flex items-center gap-2">
                  <span :class="getDifficultyColor(currentCard.difficulty)" 
                        class="px-2 py-1 rounded-lg text-xs font-semibold uppercase">
                    {{ currentCard.difficulty }}
                  </span>
                  <span v-if="currentCard.newsDate" class="text-xs text-neuron-text-secondary flex items-center gap-1">
                    <Calendar class="w-3 h-3" />
                    {{ formatNewsDate(currentCard.newsDate) }}
                  </span>
                </div>
              </div>
              
              <div class="flex-1 flex flex-col justify-center">
                <h3 class="text-xl font-semibold text-neuron-text-primary mb-4">
                  {{ currentCard.questionType || 'News Question' }}
                </h3>
                <p class="text-neuron-text-primary text-lg leading-relaxed mb-4">
                  {{ currentCard.question }}
                </p>
                <div v-if="currentCard.context" class="bg-neuron-bg-primary/50 rounded-lg p-4 border border-neuron-border/50">
                  <p class="text-sm text-neuron-text-secondary">
                    <strong>Context:</strong> {{ currentCard.context }}
                  </p>
                </div>
              </div>
              
              <div class="text-center text-neuron-text-secondary text-sm mt-6">
                Click to reveal answer
              </div>
            </div>

            <!-- Back of Card -->
            <div class="flashcard-face back absolute w-full bg-neuron-bg-content border border-neuron-border rounded-2xl p-8 min-h-[400px] flex flex-col">
              <div class="flex justify-between items-start mb-6">
                <span class="px-3 py-1 bg-accent-cyan/20 text-accent-cyan rounded-full text-sm font-medium">
                  Answer
                </span>
                <div class="flex gap-1">
                  <span v-for="tag in currentCard.tags" :key="tag" 
                        class="px-2 py-1 bg-neuron-bg-primary/50 text-neuron-text-secondary rounded-lg text-xs flex items-center gap-1">
                    <Tag class="w-3 h-3" />
                    {{ tag }}
                  </span>
                </div>
              </div>
              
              <div class="flex-1">
                <div class="text-neuron-text-primary leading-relaxed mb-6" v-html="formatAnswer(currentCard.answer)"></div>
                
                <div v-if="currentCard.source" class="bg-neuron-bg-primary/50 rounded-lg p-4 border border-neuron-border/50 mb-4">
                  <p class="text-sm text-neuron-text-secondary flex items-center gap-2">
                    <ExternalLink class="w-4 h-4" />
                    <strong>Source:</strong> {{ currentCard.source }}
                  </p>
                </div>
                
                <div v-if="currentCard.readMore" class="mb-6">
                  <a :href="currentCard.readMore" target="_blank" 
                     class="inline-flex items-center gap-2 text-accent-cyan hover:text-accent-emerald transition-colors text-sm">
                    <ExternalLink class="w-4 h-4" />
                    Read full article
                  </a>
                </div>
              </div>
              
              <!-- Card Actions -->
              <div class="flex gap-3 justify-center" @click.stop>
                <button @click="markCard('known')" 
                        class="px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-xl transition-colors flex items-center gap-2">
                  <Check class="w-4 h-4" />
                  Got it!
                </button>
                <button @click="markCard('learning')" 
                        class="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-xl transition-colors flex items-center gap-2">
                  <BookOpen class="w-4 h-4" />
                  Still learning
                </button>
                <button @click="markCard('difficult')" 
                        class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-xl transition-colors flex items-center gap-2">
                  <X class="w-4 h-4" />
                  Too difficult
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Navigation -->
        <div class="flex justify-center gap-4 mt-6">
          <button @click="previousCard" 
                  :disabled="currentIndex === 0"
                  class="px-6 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/50 transition-colors disabled:opacity-50 flex items-center gap-2">
            <ChevronLeft class="w-4 h-4" />
            Previous
          </button>
          <button @click="nextCard" 
                  :disabled="currentIndex === filteredCards.length - 1"
                  class="px-6 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/50 transition-colors disabled:opacity-50 flex items-center gap-2">
            Next
            <ChevronRight class="w-4 h-4" />
          </button>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else class="text-center py-16">
        <div class="w-20 h-20 bg-gradient-to-br from-accent-emerald/20 to-accent-cyan/20 rounded-full mx-auto flex items-center justify-center mb-6">
          <Newspaper class="w-10 h-10 text-accent-emerald" />
        </div>
        <h3 class="text-xl font-semibold text-neuron-text-primary mb-2">No flashcards available</h3>
        <p class="text-neuron-text-secondary mb-6">Generate flashcards from recent news about {{ selectedTopic }}</p>
        <button @click="generateNewsCards" 
                :disabled="loading"
                class="px-6 py-3 bg-gradient-to-r from-accent-emerald to-accent-cyan text-white rounded-xl hover:shadow-lg transition-all disabled:opacity-50">
          Generate Cards for {{ selectedTopic }}
        </button>
      </div>

      <!-- Action Buttons -->
      <div class="flex justify-center gap-4 mt-8">
        <button @click="showStats = !showStats" 
                class="px-4 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/50 transition-colors flex items-center gap-2">
          <BarChart3 class="w-4 h-4" />
          {{ showStats ? 'Hide' : 'Show' }} Statistics
        </button>
        <button @click="showSettings = !showSettings" 
                class="px-4 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/50 transition-colors flex items-center gap-2">
          <Settings class="w-4 h-4" />
          Settings
        </button>
        <button @click="exportProgress" 
                class="px-4 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/50 transition-colors flex items-center gap-2">
          <Download class="w-4 h-4" />
          Export Progress
        </button>
      </div>

      <!-- Statistics Panel -->
      <div v-if="showStats" class="mt-8 bg-neuron-bg-content/50 backdrop-blur-sm border border-neuron-border rounded-2xl p-6">
        <h3 class="text-lg font-semibold text-neuron-text-primary mb-4 flex items-center gap-2">
          <BarChart3 class="w-5 h-5 text-accent-emerald" />
          Detailed Study Statistics
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="bg-neuron-bg-primary/50 rounded-xl p-4">
            <div class="flex justify-between items-center">
              <span class="text-neuron-text-secondary">Study Streak</span>
              <span class="text-2xl font-bold text-accent-emerald">{{ detailedStats.studyStreak }} days</span>
            </div>
          </div>
          <div class="bg-neuron-bg-primary/50 rounded-xl p-4">
            <div class="flex justify-between items-center">
              <span class="text-neuron-text-secondary">Success Rate</span>
              <span class="text-2xl font-bold text-accent-cyan">{{ detailedStats.avgSuccessRate }}%</span>
            </div>
          </div>
          <div class="bg-neuron-bg-primary/50 rounded-xl p-4">
            <div class="flex justify-between items-center">
              <span class="text-neuron-text-secondary">Weekly Reviews</span>
              <span class="text-2xl font-bold text-accent-violet">{{ detailedStats.weeklyReviews }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Settings Panel -->
      <div v-if="showSettings" class="mt-8 bg-neuron-bg-content/50 backdrop-blur-sm border border-neuron-border rounded-2xl p-6">
        <h3 class="text-lg font-semibold text-neuron-text-primary mb-4 flex items-center gap-2">
          <Settings class="w-5 h-5 text-accent-emerald" />
          Settings
        </h3>
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <label class="text-neuron-text-primary">Cards per session:</label>
            <input v-model.number="settings.cardsPerSession" 
                   type="number" min="5" max="50" 
                   class="w-20 px-3 py-1 bg-neuron-bg-content border border-neuron-border rounded-lg text-neuron-text-primary focus:outline-none focus:border-accent-emerald/50">
          </div>
          <div class="flex items-center justify-between">
            <label class="text-neuron-text-primary">Auto-advance after marking:</label>
            <input v-model="settings.autoAdvance" 
                   type="checkbox" 
                   class="w-4 h-4 text-accent-emerald bg-neuron-bg-content border-neuron-border rounded focus:ring-accent-emerald">
          </div>
          <div class="flex items-center justify-between">
            <label class="text-neuron-text-primary">Default difficulty:</label>
            <select v-model="settings.defaultDifficulty" 
                    class="px-3 py-1 bg-neuron-bg-content border border-neuron-border rounded-lg text-neuron-text-primary focus:outline-none focus:border-accent-emerald/50">
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>
        </div>
        <div class="flex gap-3 mt-6">
          <button @click="saveSettings" 
                  class="px-4 py-2 bg-gradient-to-r from-accent-emerald to-accent-cyan text-white rounded-xl hover:shadow-lg transition-all">
            Save Settings
          </button>
          <button @click="showSettings = false" 
                  class="px-4 py-2 bg-neuron-bg-content border border-neuron-border rounded-xl text-neuron-text-secondary hover:text-accent-emerald transition-colors">
            Close
          </button>
        </div>
      </div>

      <!-- Recent News Preview -->
      <div v-if="recentNews.length > 0" class="mt-8 bg-neuron-bg-content/50 backdrop-blur-sm border border-neuron-border rounded-2xl p-6">
        <h3 class="text-lg font-semibold text-neuron-text-primary mb-4 flex items-center gap-2">
          <Newspaper class="w-5 h-5 text-accent-emerald" />
          Recent News Headlines
        </h3>
        <div class="grid gap-4">
          <div v-for="article in recentNews" :key="article.id" 
               class="bg-neuron-bg-primary/50 rounded-xl p-4 border border-neuron-border/50 hover:border-accent-emerald/30 transition-colors">
            <div class="flex justify-between items-start gap-4">
              <div class="flex-1">
                <h4 class="text-neuron-text-primary font-semibold mb-2">{{ article.title }}</h4>
                <p class="text-neuron-text-secondary text-sm mb-3">{{ article.summary }}</p>
                <div class="flex items-center gap-4 text-xs text-neuron-text-secondary">
                  <span>{{ article.source }}</span>
                  <span>{{ formatNewsDate(article.publishedDate) }}</span>
                </div>
              </div>
              <button @click="generateFromArticle(article)" 
                      class="px-3 py-1 bg-accent-emerald/20 text-accent-emerald rounded-lg hover:bg-accent-emerald hover:text-white transition-colors text-sm flex items-center gap-1">
                <Plus class="w-3 h-3" />
                Create Cards
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed, onMounted, ref, watch } from 'vue'
import { 
  Brain, Newspaper, Shuffle, Loader2, BookOpen, Calendar, 
  Tag, ExternalLink, Check, X, ChevronLeft, ChevronRight, 
  BarChart3, Settings, Download, Plus 
} from 'lucide-vue-next'

export default {
  name: 'NewsFlashcards',
  components: {
    Brain, Newspaper, Shuffle, Loader2, BookOpen, Calendar, 
    Tag, ExternalLink, Check, X, ChevronLeft, ChevronRight, 
    BarChart3, Settings, Download, Plus
  },
  setup() {
    // Reactive state
    const flashcards = ref([])
    const currentIndex = ref(0)
    const isFlipped = ref(false)
    const selectedCategory = ref('all')
    const selectedTopic = ref('politics')
    const sourceType = ref('recent_news')
    const loading = ref(false)
    const studyMode = ref(false)
    const showStats = ref(false)
    const showSettings = ref(false)
    const sessionStats = ref({ correct: 0, incorrect: 0, total: 0 })
    const detailedStats = ref({ studyStreak: 7, avgSuccessRate: 85, weeklyReviews: 42 })
    const recentNews = ref([])
    const settings = ref({
      cardsPerSession: 20,
      autoAdvance: true,
      defaultDifficulty: 'medium'
    })

    const newsTopics = [
      'politics', 'technology', 'climate change', 'economy', 
      'health', 'sports', 'world news', 'business', 'science'
    ]

    // Sample news flashcards
    const sampleNewsCards = [
      {
        id: '1',
        category: 'Current Events - Politics',
        question: 'What recent political development has been making headlines regarding climate policy?',
        answer: `Recent developments in climate policy include new international agreements and domestic policy changes. Key points include carbon emission targets, renewable energy investments, and international cooperation frameworks.`,
        difficulty: 'medium',
        tags: ['politics', 'climate', 'policy'],
        status: 'new',
        questionType: 'Current Affairs',
        newsDate: new Date().toISOString(),
        source: 'Associated Press',
        context: 'Global climate summit discussions',
        readMore: 'https://example.com/climate-policy-news'
      },
      {
        id: '2', 
        category: 'Technology News',
        question: 'What major AI development has been announced recently?',
        answer: `Recent AI developments include advances in large language models, new applications in healthcare, and regulatory frameworks being proposed. Companies are focusing on AI safety and ethical deployment.`,
        difficulty: 'medium',
        tags: ['technology', 'ai', 'innovation'],
        status: 'new',
        questionType: 'Tech Update',
        newsDate: new Date().toISOString(),
        source: 'TechCrunch',
        context: 'AI industry developments',
        readMore: 'https://example.com/ai-development-news'
      },
      {
        id: '3',
        category: 'Economic News',
        question: 'What are the current trends in global markets?',
        answer: `Current market trends show mixed signals with inflation concerns, central bank policies affecting interest rates, and sector-specific performance variations. Emerging markets are showing resilience while developed markets face challenges.`,
        difficulty: 'hard',
        tags: ['economy', 'markets', 'finance'],
        status: 'new',
        questionType: 'Market Analysis',
        newsDate: new Date().toISOString(),
        source: 'Financial Times',
        context: 'Global economic overview',
        readMore: 'https://example.com/market-trends-news'
      },
      {
        id: '4',
        category: 'Health News',
        question: 'What recent health breakthrough has been reported in medical research?',
        answer: `Recent medical breakthroughs include new treatments for chronic diseases, advances in personalized medicine, and innovative diagnostic technologies. Research focuses on precision therapy and preventive care approaches.`,
        difficulty: 'medium',
        tags: ['health', 'medical', 'research'],
        status: 'new',
        questionType: 'Health Update',
        newsDate: new Date(Date.now() - 24*60*60*1000).toISOString(),
        source: 'Medical News Today',
        context: 'Latest medical research findings'
      },
      {
        id: '5',
        category: 'Climate News',
        question: 'What environmental milestone or concern has been highlighted recently?',
        answer: `Recent environmental news covers climate change impacts, renewable energy milestones, biodiversity conservation efforts, and sustainable development initiatives. Focus areas include carbon reduction strategies and green technology adoption.`,
        difficulty: 'medium',
        tags: ['climate', 'environment', 'sustainability'],
        status: 'new',
        questionType: 'Environmental Update',
        newsDate: new Date(Date.now() - 48*60*60*1000).toISOString(),
        source: 'Environmental News Network',
        context: 'Global environmental developments'
      }
    ]

    // Computed properties
    const filteredCards = computed(() => {
      if (selectedCategory.value === 'all') {
        return flashcards.value
      }
      return flashcards.value.filter(card => card.category === selectedCategory.value)
    })

    const categories = computed(() => {
      const cats = [...new Set(flashcards.value.map(card => card.category))]
      return cats.sort()
    })

    const currentCard = computed(() => {
      return filteredCards.value[currentIndex.value] || {}
    })

    const progressPercentage = computed(() => {
      if (filteredCards.value.length === 0) return 0
      return ((currentIndex.value + 1) / filteredCards.value.length) * 100
    })

    const stats = computed(() => {
      const known = flashcards.value.filter(card => card.status === 'known').length
      const learning = flashcards.value.filter(card => card.status === 'learning').length
      const dueToday = flashcards.value.filter(card => {
        if (!card.nextReview) return card.status === 'new'
        return new Date(card.nextReview) <= new Date()
      }).length
      
      return {
        total: { value: flashcards.value.length, label: 'Total Cards', icon: '📚', color: 'bg-blue-500/20' },
        known: { value: known, label: 'Mastered', icon: '✅', color: 'bg-green-500/20' },
        learning: { value: learning, label: 'Learning', icon: '🎯', color: 'bg-yellow-500/20' },
        dueToday: { value: dueToday, label: 'Due Today', icon: '⏰', color: 'bg-red-500/20' }
      }
    })

    // Methods
    const getTopicEmoji = (topic) => {
      const emojiMap = {
        'politics': '🏛️',
        'technology': '💻', 
        'climate change': '🌍',
        'economy': '📈',
        'health': '🏥',
        'sports': '⚽',
        'world news': '🌐',
        'business': '💼',
        'science': '🔬'
      }
      return emojiMap[topic] || '📰'
    }

    const getDifficultyColor = (difficulty) => {
      const colorMap = {
        'easy': 'bg-green-500/20 text-green-400',
        'medium': 'bg-yellow-500/20 text-yellow-400',
        'hard': 'bg-red-500/20 text-red-400'
      }
      return colorMap[difficulty] || colorMap['medium']
    }

    const formatNewsDate = (dateString) => {
      if (!dateString) return ''
      const date = new Date(dateString)
      const now = new Date()
      const diffTime = Math.abs(now - date)
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
      
      if (diffDays === 1) return 'Today'
      if (diffDays === 2) return 'Yesterday'
      if (diffDays < 7) return `${diffDays} days ago`
      return date.toLocaleDateString()
    }

    const formatAnswer = (answer) => {
      return answer.replace(/\n/g, '<br/>')
    }

    const flipCard = () => {
      isFlipped.value = !isFlipped.value
    }

    const nextCard = () => {
      if (currentIndex.value < filteredCards.value.length - 1) {
        currentIndex.value++
        isFlipped.value = false
      }
    }

    const previousCard = () => {
      if (currentIndex.value > 0) {
        currentIndex.value--
        isFlipped.value = false
      }
    }

    const shuffleCards = () => {
      const shuffled = [...flashcards.value].sort(() => Math.random() - 0.5)
      flashcards.value = shuffled
      currentIndex.value = 0
      isFlipped.value = false
    }

    const markCard = (status) => {
      const card = currentCard.value
      card.status = status
      
      if (studyMode.value) {
        sessionStats.value.total++
        if (status === 'known') {
          sessionStats.value.correct++
        } else {
          sessionStats.value.incorrect++
        }
      }
      
      updateStats()
      recordReview(card.id, status === 'known')
      
      // Auto advance to next card if enabled
      if (settings.value.autoAdvance) {
        setTimeout(() => {
          if (currentIndex.value < filteredCards.value.length - 1) {
            nextCard()
          } else {
            if (studyMode.value) {
              endStudySession()
            }
          }
        }, 500)
      }
    }

    const updateStats = () => {
      // Stats are now computed automatically
    }

    const filterCards = () => {
      currentIndex.value = 0
      isFlipped.value = false
    }

    const startStudySession = () => {
      studyMode.value = true
      sessionStats.value = { correct: 0, incorrect: 0, total: 0 }
      
      // Filter to cards due for review or new cards
      const dueCards = flashcards.value.filter(card => {
        if (card.status === 'new') return true
        if (!card.nextReview) return true
        return new Date(card.nextReview) <= new Date()
      })
      
      if (dueCards.length > 0) {
        flashcards.value = dueCards.slice(0, settings.value.cardsPerSession)
        currentIndex.value = 0
        isFlipped.value = false
      } else {
        alert('No cards are due for review right now!')
        studyMode.value = false
      }
    }

    const endStudySession = () => {
      studyMode.value = false
      const accuracy = sessionStats.value.total > 0 
        ? Math.round((sessionStats.value.correct / sessionStats.value.total) * 100)
        : 0
      
      alert(`Study session complete!\n✅ Correct: ${sessionStats.value.correct}\n❌ Incorrect: ${sessionStats.value.incorrect}\nAccuracy: ${accuracy}%`)
      loadAllCards()
    }

    const loadAllCards = () => {
      flashcards.value = [...sampleNewsCards]
    }

    const saveSettings = () => {
      localStorage.setItem('newsFlashcardSettings', JSON.stringify(settings.value))
      alert('Settings saved!')
    }

    const loadSettings = () => {
      const saved = localStorage.getItem('newsFlashcardSettings')
      if (saved) {
        settings.value = { ...settings.value, ...JSON.parse(saved) }
      }
    }

    const exportProgress = () => {
      const progressData = {
        flashcards: flashcards.value,
        stats: stats.value,
        exportDate: new Date().toISOString()
      }
      
      const blob = new Blob([JSON.stringify(progressData, null, 2)], {
        type: 'application/json'
      })
      
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `news-flashcards-progress-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }

    // API integration
    const generateNewsCards = async () => {
      loading.value = true
      try {
        const response = await fetch('http://localhost:8000/flashcards/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            topic: selectedTopic.value,
            source_type: sourceType.value,
            count: 10,
            difficulty: settings.value.defaultDifficulty
          })
        })
        
        if (!response.ok) {
          throw new Error('Failed to generate flashcards')
        }
        
        const data = await response.json()
        
        if (data.success) {
          flashcards.value = [...flashcards.value, ...data.flashcards]
          alert(`Generated ${data.flashcards.length} flashcards about ${selectedTopic.value}!`)
        } else {
          throw new Error(data.error || 'Generation failed')
        }
        
      } catch (error) {
        console.error('Failed to generate cards:', error)
        
        // Fallback to demo generation
        const newCards = sampleNewsCards.map((card, index) => ({
          ...card,
          id: card.id + '_new_' + Date.now() + index,
          question: card.question + ' (Demo)',
        }))
        
        flashcards.value = [...flashcards.value, ...newCards.slice(0, 3)]
        alert(`Generated ${newCards.slice(0, 3).length} demo flashcards about ${selectedTopic.value}!`)
      } finally {
        loading.value = false
      }
    }

    const recordReview = async (cardId, correct) => {
      try {
        await fetch(`http://localhost:8000/flashcards/${cardId}/review`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            correct: correct,
            difficulty_rating: correct ? 3 : 4
          })
        })
      } catch (error) {
        console.error('Failed to record review:', error)
      }
    }

    const fetchRecentNews = async () => {
      try {
        const mockNews = [
          {
            id: 'news1',
            title: 'Climate Summit Reaches New Agreement on Carbon Emissions',
            summary: 'World leaders agree on ambitious new carbon reduction targets for 2030...',
            source: 'Reuters',
            publishedDate: new Date().toISOString()
          },
          {
            id: 'news2', 
            title: 'Tech Industry Announces Major AI Safety Initiative',
            summary: 'Leading technology companies collaborate on comprehensive AI safety standards...',
            source: 'TechCrunch',
            publishedDate: new Date(Date.now() - 24*60*60*1000).toISOString()
          },
          {
            id: 'news3',
            title: 'Global Markets Show Mixed Signals Amid Economic Uncertainty',
            summary: 'Stock markets fluctuate as investors weigh inflation concerns against growth prospects...',
            source: 'Financial Times',
            publishedDate: new Date(Date.now() - 48*60*60*1000).toISOString()
          }
        ]
        recentNews.value = mockNews
      } catch (error) {
        console.error('Failed to fetch recent news:', error)
      }
    }

    const generateFromArticle = async (article) => {
      loading.value = true
      try {
        const response = await fetch('http://localhost:8000/flashcards/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            topic: article.title,
            source_type: 'specific_article',
            article_data: article,
            count: 5,
            difficulty: settings.value.defaultDifficulty
          })
        })
        
        if (!response.ok) {
          throw new Error('Failed to generate from article')
        }
        
        const data = await response.json()
        if (data.success) {
          flashcards.value = [...flashcards.value, ...data.flashcards]
          alert(`Generated ${data.flashcards.length} cards from the article!`)
        }
      } catch (error) {
        console.error('Failed to generate from article:', error)
        
        // Fallback demo generation
        const newCard = {
          id: 'article_' + article.id + '_' + Date.now(),
          category: 'Generated from Article',
          question: `What are the key points from: "${article.title}"?`,
          answer: article.summary + ' (Generated from article content with additional analysis and context)',
          difficulty: 'medium',
          tags: ['generated', 'news'],
          status: 'new',
          questionType: 'Article Analysis',
          newsDate: article.publishedDate,
          source: article.source,
          context: 'Generated from news article'
        }
        
        flashcards.value = [...flashcards.value, newCard]
        alert('Generated 1 demo card from the article!')
      } finally {
        loading.value = false
      }
    }

    const fetchDetailedStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/flashcards/stats')
        if (response.ok) {
          const data = await response.json()
          if (data.success) {
            detailedStats.value = {
              studyStreak: data.study_streak_days || 7,
              avgSuccessRate: Math.round((data.overall_stats?.average_success_rate || 0.85) * 100),
              weeklyReviews: data.recent_performance?.reduce((sum, day) => sum + day.cards_reviewed, 0) || 42
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch detailed stats:', error)
        // Using demo values as fallback
      }
    }

    const getParticleColor = (index) => {
      const colors = [
        'bg-accent-emerald/20',
        'bg-accent-cyan/20',
        'bg-accent-violet/20',
        'bg-accent-rose/20',
        'bg-accent-amber/20',
        'bg-accent-orange/20'
      ]
      return colors[index % colors.length]
    }

    const particleStyle = (index) => {
      return {
        left: `${(index * 13 + 10) % 100}%`,
        top: `${(index * 17 + 20) % 100}%`,
        animationDelay: `${index * 0.5}s`,
        animationDuration: `${3 + (index % 3)}s`
      }
    }

    // Watch for topic changes
    watch(selectedTopic, () => {
      fetchRecentNews()
    })

    // Lifecycle
    onMounted(() => {
      loadAllCards()
      fetchRecentNews()
      fetchDetailedStats()
      loadSettings()
    })

    return {
      // State
      flashcards,
      currentIndex,
      isFlipped,
      selectedCategory,
      selectedTopic,
      sourceType,
      loading,
      studyMode,
      showStats,
      showSettings,
      sessionStats,
      detailedStats,
      recentNews,
      newsTopics,
      settings,
      
      // Computed
      filteredCards,
      categories,
      currentCard,
      progressPercentage,
      stats,
      
      // Methods
      getTopicEmoji,
      getDifficultyColor,
      formatNewsDate,
      formatAnswer,
      flipCard,
      nextCard,
      previousCard,
      shuffleCards,
      markCard,
      filterCards,
      startStudySession,
      generateNewsCards,
      generateFromArticle,
      saveSettings,
      exportProgress,
      getParticleColor,
      particleStyle
    }
  }
}
</script>

<style scoped>
/* Neuron theme colors */
.bg-neuron-bg-primary { background-color: #0f172a; }
.bg-neuron-bg-content { background-color: #1e293b; }
.text-neuron-text-primary { color: #f1f5f9; }
.text-neuron-text-secondary { color: #94a3b8; }
.border-neuron-border { border-color: #334155; }

/* Accent colors */
.bg-accent-emerald { background-color: #10b981; }
.bg-accent-cyan { background-color: #06b6d4; }
.bg-accent-violet { background-color: #8b5cf6; }
.bg-accent-rose { background-color: #f43f5e; }
.bg-accent-amber { background-color: #f59e0b; }
.bg-accent-orange { background-color: #f97316; }

.text-accent-emerald { color: #10b981; }
.text-accent-cyan { color: #06b6d4; }
.text-accent-violet { color: #8b5cf6; }

.border-accent-emerald\/50 { border-color: rgba(16, 185, 129, 0.5); }
.border-accent-emerald\/30 { border-color: rgba(16, 185, 129, 0.3); }

.hover\:text-accent-emerald:hover { color: #10b981; }
.hover\:border-accent-emerald\/50:hover { border-color: rgba(16, 185, 129, 0.5); }

/* Custom flashcard 3D flip animation */
.flashcard-container {
  perspective: 1000px;
}

.flashcard-inner {
  position: relative;
  width: 100%;
  height: 100%;
  transition: transform 0.7s;
  transform-style: preserve-3d;
}

.flashcard-inner.flipped {
  transform: rotateY(180deg);
}

.flashcard-face {
  backface-visibility: hidden;
}

.flashcard-face.back {
  transform: rotateY(180deg);
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(156, 163, 175, 0.3);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(156, 163, 175, 0.5);
}

/* Smooth animations */
.transition-all {
  transition: all 0.3s ease;
}

/* Custom checkbox */
input[type="checkbox"] {
  accent-color: #10b981;
}

/* Backdrop blur support */
.backdrop-blur-sm {
  backdrop-filter: blur(4px);
}

/* Gradient backgrounds */
.from-accent-emerald { --tw-gradient-from: #10b981; }
.to-accent-cyan { --tw-gradient-to: #06b6d4; }
.from-accent-violet { --tw-gradient-from: #8b5cf6; }
.to-accent-rose { --tw-gradient-to: #f43f5e; }

/* Hover effects */
.hover:shadow-lg:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

/* Animation keyframes */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.animate-pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.animate-spin {
  animation: spin 1s linear infinite;
}

/* Additional responsive utilities */
@media (max-width: 640px) {
  .flashcard-face {
    min-height: 350px;
    padding: 1.5rem;
  }
  
  .grid {
    grid-template-columns: repeat(1, minmax(0, 1fr));
  }
  
  .lg\:grid-cols-4 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 768px) {
  .md\:grid-cols-3 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

/* Focus styles for accessibility */
button:focus,
select:focus,
input:focus {
  outline: 2px solid #10b981;
  outline-offset: 2px;
}

/* Loading state animations */
.loading-shimmer {
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

/* Card flip enhancement */
.flashcard-face {
  cursor: pointer;
  transition: all 0.3s ease;
}

.flashcard-face:hover {
  transform: translateY(-2px);
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

.flashcard-inner.flipped .flashcard-face:hover {
  transform: rotateY(180deg) translateY(-2px);
}

/* Particle animations */
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

.float-animation {
  animation: float 6s ease-in-out infinite;
}

/* Progress bar animation */
.progress-fill {
  transition: width 0.5s ease-in-out;
}

/* Button hover effects */
.btn-hover-scale {
  transition: transform 0.2s ease-in-out;
}

.btn-hover-scale:hover {
  transform: scale(1.05);
}

/* Text gradient utility */
.text-gradient {
  background: linear-gradient(135deg, #10b981, #06b6d4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Card status indicators */
.status-known {
  border-left: 4px solid #10b981;
}

.status-learning {
  border-left: 4px solid #f59e0b;
}

.status-difficult {
  border-left: 4px solid #ef4444;
}

.status-new {
  border-left: 4px solid #6b7280;
}

/* Enhanced backdrop blur for better browser support */
@supports (backdrop-filter: blur(4px)) {
  .backdrop-blur-sm {
    backdrop-filter: blur(4px);
  }
}

@supports not (backdrop-filter: blur(4px)) {
  .backdrop-blur-sm {
    background-color: rgba(30, 41, 59, 0.8);
  }
}

/* Enhanced gradient backgrounds */
.gradient-bg-primary {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

.gradient-bg-secondary {
  background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
}

/* Enhanced shadow utilities */
.shadow-neuron {
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
}

.shadow-neuron-lg {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
}

/* Custom selection colors */
::selection {
  background-color: rgba(16, 185, 129, 0.3);
  color: #f1f5f9;
}

/* Enhanced form elements */
select, input[type="number"] {
  appearance: none;
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
  background-position: right 0.5rem center;
  background-repeat: no-repeat;
  background-size: 1.5em 1.5em;
  padding-right: 2.5rem;
}

/* Dark mode enhancements */
@media (prefers-color-scheme: dark) {
  .news-flashcard-app {
    color-scheme: dark;
  }
}

/* Print styles */
@media print {
  .flashcard-face {
    break-inside: avoid;
    background: white !important;
    color: black !important;
    border: 1px solid black !important;
  }
  
  .backdrop-blur-sm,
  .bg-neuron-bg-primary,
  .bg-neuron-bg-content {
    background: white !important;
  }
}
</style>