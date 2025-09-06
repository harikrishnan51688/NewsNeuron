<template>
  <div class="news-flashcard-app">
    <!-- Header -->
    <div class="header">
      <h1>📰 NewsNeuron Flashcards</h1>
      <p>Stay informed with AI-powered news learning</p>
    </div>

    <!-- News Topics Selection -->
    <div class="news-topics">
      <h3>Select News Topic</h3>
      <div class="topic-grid">
        <button 
          v-for="topic in newsTopics" 
          :key="topic"
          @click="selectedTopic = topic"
          :class="['topic-btn', { active: selectedTopic === topic }]"
        >
          {{ getTopicEmoji(topic) }} {{ topic }}
        </button>
      </div>
    </div>

    <!-- Stats Dashboard -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon">📚</div>
        <div class="stat-info">
          <h3>{{ stats.total }}</h3>
          <p>Total Cards</p>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">✅</div>
        <div class="stat-info">
          <h3>{{ stats.known }}</h3>
          <p>Mastered</p>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-info">
          <h3>{{ stats.learning }}</h3>
          <p>Learning</p>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">⏰</div>
        <div class="stat-info">
          <h3>{{ stats.dueToday }}</h3>
          <p>Due Today</p>
        </div>
      </div>
    </div>

    <!-- Controls -->
    <div class="controls">
      <select v-model="selectedCategory" @change="filterCards" class="category-select">
        <option value="all">All Categories</option>
        <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
      </select>
      
      <select v-model="sourceType" class="source-select">
        <option value="recent_news">Recent News</option>
        <option value="knowledge_base">Stored Articles</option>
        <option value="entities">Knowledge Graph</option>
      </select>
      
      <button @click="shuffleCards" class="btn btn-secondary">
        🔀 Shuffle
      </button>
      <button @click="generateNewsCards" class="btn btn-primary" :disabled="loading">
        {{ loading ? '⏳ Generating...' : '📰 Generate News Cards' }}
      </button>
      <button @click="startStudySession" class="btn btn-success">
        🧠 Start Study Session
      </button>
    </div>

    <!-- Study Session Info -->
    <div v-if="studyMode" class="study-session">
      <div class="session-header">
        <h3>📖 Study Session Active</h3>
        <div class="session-stats">
          <span class="correct">✅ {{ sessionStats.correct }}</span>
          <span class="incorrect">❌ {{ sessionStats.incorrect }}</span>
          <span class="total">Total: {{ sessionStats.total }}</span>
        </div>
      </div>
    </div>

    <!-- Flashcard Container -->
    <div class="flashcard-container" v-if="filteredCards.length > 0">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progressPercentage + '%' }"></div>
      </div>
      
      <div class="card-counter">
        {{ currentIndex + 1 }} / {{ filteredCards.length }}
      </div>

      <div class="flashcard" :class="{ 'flipped': isFlipped }" @click="flipCard">
        <div class="card-face card-front">
          <div class="card-header">
            <span class="category-badge">{{ currentCard.category }}</span>
            <div class="card-meta">
              <span class="difficulty-badge" :class="currentCard.difficulty">
                {{ currentCard.difficulty }}
              </span>
              <span v-if="currentCard.newsDate" class="news-date">
                📅 {{ formatNewsDate(currentCard.newsDate) }}
              </span>
            </div>
          </div>
          <div class="card-content">
            <h3>{{ currentCard.questionType || 'News Question' }}</h3>
            <p class="question-text">{{ currentCard.question }}</p>
            <div v-if="currentCard.context" class="context">
              <small><strong>Context:</strong> {{ currentCard.context }}</small>
            </div>
          </div>
          <div class="card-footer">
            <small>Click to reveal answer</small>
          </div>
        </div>

        <div class="card-face card-back">
          <div class="card-header">
            <span class="category-badge">{{ currentCard.category }}</span>
            <div class="tags">
              <span v-for="tag in currentCard.tags" :key="tag" class="tag">{{ tag }}</span>
            </div>
          </div>
          <div class="card-content">
            <h3>Answer</h3>
            <div class="answer-content" v-html="formatAnswer(currentCard.answer)"></div>
            <div v-if="currentCard.source" class="source-info">
              <small><strong>Source:</strong> {{ currentCard.source }}</small>
            </div>
            <div v-if="currentCard.readMore" class="read-more">
              <a :href="currentCard.readMore" target="_blank" rel="noopener noreferrer">
                📖 Read full article
              </a>
            </div>
          </div>
          <div class="card-actions">
            <button @click.stop="markCard('known')" class="btn btn-success">
              ✅ Got it!
            </button>
            <button @click.stop="markCard('learning')" class="btn btn-warning">
              📚 Still learning
            </button>
            <button @click.stop="markCard('difficult')" class="btn btn-danger">
              😵 Too difficult
            </button>
          </div>
        </div>
      </div>

      <!-- Navigation -->
      <div class="navigation">
        <button @click="previousCard" :disabled="currentIndex === 0" class="btn btn-nav">
          ← Previous
        </button>
        <button @click="nextCard" :disabled="currentIndex === filteredCards.length - 1" class="btn btn-nav">
          Next →
        </button>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else class="empty-state">
      <div class="empty-icon">📰</div>
      <h3>No news flashcards available</h3>
      <p>Generate flashcards from recent news about {{ selectedTopic }}</p>
      <button @click="generateNewsCards" class="btn btn-primary" :disabled="loading">
        📰 Generate Cards for {{ selectedTopic }}
      </button>
    </div>

    <!-- Recent News Preview -->
    <div v-if="recentNews.length > 0" class="recent-news">
      <h3>Recent News Headlines</h3>
      <div class="news-grid">
        <div v-for="article in recentNews" :key="article.id" class="news-item">
          <div class="news-content">
            <h4>{{ article.title }}</h4>
            <p>{{ article.summary }}</p>
            <div class="news-meta">
              <span class="news-source">{{ article.source }}</span>
              <span class="news-date">{{ formatNewsDate(article.publishedDate) }}</span>
            </div>
          </div>
          <button @click="generateFromArticle(article)" class="btn btn-small">
            ➕ Create Cards
          </button>
        </div>
      </div>
    </div>

    <!-- Study Statistics -->
    <div v-if="showStats" class="study-statistics">
      <h3>Study Statistics</h3>
      <div class="stats-detail">
        <div class="stat-row">
          <span>Study Streak:</span>
          <span class="stat-value">{{ detailedStats.studyStreak }} days</span>
        </div>
        <div class="stat-row">
          <span>Average Success Rate:</span>
          <span class="stat-value">{{ detailedStats.avgSuccessRate }}%</span>
        </div>
        <div class="stat-row">
          <span>Cards Reviewed This Week:</span>
          <span class="stat-value">{{ detailedStats.weeklyReviews }}</span>
        </div>
      </div>
      <button @click="showStats = false" class="btn btn-secondary">Hide Stats</button>
    </div>

    <!-- Settings Panel -->
    <div v-if="showSettings" class="settings-panel">
      <h3>Settings</h3>
      <div class="setting-item">
        <label>Cards per session:</label>
        <input v-model.number="settings.cardsPerSession" type="number" min="5" max="50" />
      </div>
      <div class="setting-item">
        <label>Auto-advance after marking:</label>
        <input v-model="settings.autoAdvance" type="checkbox" />
      </div>
      <div class="setting-item">
        <label>Default difficulty:</label>
        <select v-model="settings.defaultDifficulty">
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>
      <button @click="saveSettings" class="btn btn-primary">Save Settings</button>
      <button @click="showSettings = false" class="btn btn-secondary">Close</button>
    </div>

    <!-- Action Buttons -->
    <div class="action-buttons">
      <button @click="showStats = !showStats" class="btn btn-info">
        📊 {{ showStats ? 'Hide' : 'Show' }} Statistics
      </button>
      <button @click="showSettings = !showSettings" class="btn btn-info">
        ⚙️ Settings
      </button>
      <button @click="exportProgress" class="btn btn-secondary">
        📤 Export Progress
      </button>
    </div>
  </div>
</template>

<script>
import { computed, onMounted, ref, watch } from 'vue'
export default {
  name: 'FlashcardsView',
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
    const stats = ref({ known: 0, learning: 0, total: 0, dueToday: 0 })
    const sessionStats = ref({ correct: 0, incorrect: 0, total: 0 })
    const detailedStats = ref({ studyStreak: 0, avgSuccessRate: 0, weeklyReviews: 0 })
    const recentNews = ref([])
    const settings = ref({
      cardsPerSession: 20,
      autoAdvance: true,
      defaultDifficulty: 'medium'
    })

    const newsTopics = ref([
      'politics', 'technology', 'climate change', 'economy', 
      'health', 'sports', 'world news', 'business', 'science'
    ])

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
      const known = flashcards.value.filter(card => card.status === 'known').length
      const learning = flashcards.value.filter(card => card.status === 'learning').length
      const dueToday = flashcards.value.filter(card => {
        if (!card.nextReview) return card.status === 'new'
        return new Date(card.nextReview) <= new Date()
      }).length
      
      stats.value = {
        known,
        learning,
        total: flashcards.value.length,
        dueToday
      }
    }

    const filterCards = () => {
      currentIndex.value = 0
      isFlipped.value = false
    }

    const formatAnswer = (answer) => {
      return answer.replace(/\n/g, '<br/>')
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
      updateStats()
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
          updateStats()
          alert(`Generated ${data.flashcards.length} flashcards about ${selectedTopic.value}!`)
        } else {
          throw new Error(data.error || 'Generation failed')
        }
        
      } catch (error) {
        console.error('Failed to generate cards:', error)
        alert('Failed to generate flashcards. Please try again.')
      } finally {
        loading.value = false
      }
    }

    const recordReview = async (cardId, correct) => {
      try {
        await fetch('http://localhost:8000/flashcards/${cardId}/review', {
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
        
        const data = await response.json()
        if (data.success) {
          flashcards.value = [...flashcards.value, ...data.flashcards]
          updateStats()
          alert(`Generated ${data.flashcards.length} cards from the article!`)
        }
      } catch (error) {
        console.error('Failed to generate from article:', error)
        alert('Failed to generate cards from article.')
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
              studyStreak: data.study_streak_days || 0,
              avgSuccessRate: Math.round((data.overall_stats?.average_success_rate || 0) * 100),
              weeklyReviews: data.recent_performance?.reduce((sum, day) => sum + day.cards_reviewed, 0) || 0
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch detailed stats:', error)
      }
    }

    // Watch for topic changes
    watch(selectedTopic, () => {
      fetchRecentNews()
    })

    // Lifecycle
 onMounted(() => {
  generateNewsCards()   // fetches flashcards from backend
  fetchRecentNews()     // pulls headlines from /news/recent
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
      stats,
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
      
      // Methods
      getTopicEmoji,
      formatNewsDate,
      flipCard,
      nextCard,
      previousCard,
      shuffleCards,
      markCard,
      filterCards,
      formatAnswer,
      startStudySession,
      generateNewsCards,
      generateFromArticle,
      saveSettings,
      exportProgress
    }
  }
}
</script>

<style scoped>
.news-flashcard-app {
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f8fafc;
  min-height: 100vh;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  color: #1f2937;
  font-size: 2.5rem;
  margin-bottom: 8px;
  font-weight: 700;
}

.header p {
  color: #6b7280;
  font-size: 1.1rem;
}

.news-topics {
  margin-bottom: 30px;
}

.news-topics h3 {
  color: #1f2937;
  margin-bottom: 15px;
  text-align: center;
  font-size: 1.2rem;
}

.topic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.topic-btn {
  padding: 12px 16px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
  font-weight: 500;
  text-transform: capitalize;
}

.topic-btn:hover {
  border-color: #3b82f6;
  background: #f0f9ff;
}

.topic-btn.active {
  background: #3b82f6;
  color: white;
  border-color: #3b82f6;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.stat-card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  gap: 15px;
  border: 1px solid #e5e7eb;
}

.stat-icon {
  font-size: 2rem;
}

.stat-info h3 {
  font-size: 1.8rem;
  font-weight: bold;
  color: #1f2937;
  margin: 0;
}

.stat-info p {
  color: #6b7280;
  margin: 0;
  font-size: 0.9rem;
}

.controls {
  display: flex;
  gap: 15px;
  margin-bottom: 30px;
  justify-content: center;
  flex-wrap: wrap;
}

.category-select {
  padding: 10px 15px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  background: white;
  font-size: 14px;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
  font-size: 14px;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
}

.btn-secondary {
  background: white;
  color: #3b82f6;
  border: 2px solid #3b82f6;
}

.btn-secondary:hover {
  background: #3b82f6;
  color: white;
}

.btn-success {
  background: #10b981;
  color: white;
}

.btn-warning {
  background: #f59e0b;
  color: white;
}

.btn-nav {
  background: #f3f4f6;
  color: #374151;
}

.flashcard-container {
  text-align: center;
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  margin-bottom: 15px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #3b82f6;
  transition: width 0.3s ease;
}

.card-counter {
  color: #6b7280;
  margin-bottom: 20px;
  font-size: 14px;
}

.flashcard {
  width: 100%;
  height: 400px;
  perspective: 1000px;
  margin-bottom: 30px;
  cursor: pointer;
}

.card-face {
  width: 100%;
  height: 100%;
  position: absolute;
  backface-visibility: hidden;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  background: white;
  padding: 30px;
  display: flex;
  flex-direction: column;
  transition: transform 0.6s ease;
}

.card-back {
  transform: rotateY(180deg);
}

.flashcard.flipped .card-front {
  transform: rotateY(180deg);
}

.flashcard.flipped .card-back {
  transform: rotateY(0);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.category-badge {
  background: #dbeafe;
  color: #1d4ed8;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}

.difficulty-badge {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}

.difficulty-badge.easy { background: #dcfce7; color: #166534; }
.difficulty-badge.medium { background: #fef3c7; color: #92400e; }
.difficulty-badge.hard { background: #fecaca; color: #991b1b; }

.card-content {
  flex: 1;
  text-align: left;
}

.card-content h3 {
  color: #1f2937;
  margin-bottom: 15px;
  font-size: 1.2rem;
}

.card-content p {
  color: #374151;
  line-height: 1.6;
  font-size: 1rem;
}

.answer-content {
  color: #374151;
  line-height: 1.6;
}

.answer-content code {
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 0.9em;
}

.card-footer {
  color: #9ca3af;
  font-size: 12px;
  text-align: center;
}

.card-actions {
  display: flex;
  gap: 15px;
  justify-content: center;
  margin-top: 20px;
}

.tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.tag {
  background: #f3f4f6;
  color: #374151;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
}

.navigation {
  display: flex;
  gap: 20px;
  justify-content: center;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 20px;
}

.empty-state h3 {
  color: #1f2937;
  margin-bottom: 10px;
}

.empty-state p {
  color: #6b7280;
  margin-bottom: 30px;
}
</style>