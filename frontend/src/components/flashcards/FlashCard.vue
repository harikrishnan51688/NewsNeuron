<template>
  <div class="flashcard-app">
    <!-- Header -->
    <div class="header">
      <h1>📰 NewsNeuron Flashcards</h1>
      <p>Master your news AI application knowledge</p>
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
    </div>

    <!-- Controls -->
    <div class="controls">
      <select v-model="selectedCategory" @change="filterCards" class="category-select">
        <option value="all">All Categories</option>
        <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
      </select>
      <button @click="shuffleCards" class="btn btn-secondary">
        🔀 Shuffle
      </button>
      <button @click="generateFromAPI" class="btn btn-primary" :disabled="loading">
        {{ loading ? '⏳ Generating...' : '🤖 Generate from API' }}
      </button>
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
            <span class="difficulty-badge" :class="currentCard.difficulty">
              {{ currentCard.difficulty }}
            </span>
          </div>
          <div class="card-content">
            <h3>Question</h3>
            <p>{{ currentCard.question }}</p>
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
          </div>
          <div class="card-actions">
            <button @click.stop="markCard('known')" class="btn btn-success">
              ✅ Got it!
            </button>
            <button @click.stop="markCard('learning')" class="btn btn-warning">
              📚 Still learning
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
      <div class="empty-icon">📝</div>
      <h3>No flashcards available</h3>
      <p>Generate some cards from your NewsNeuron API or add them manually</p>
      <button @click="generateFromAPI" class="btn btn-primary">
        🤖 Generate Cards
      </button>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'

export default {
  name: 'NewsNeuronFlashcards',
  setup() {
    // Reactive state
    const flashcards = ref([])
    const currentIndex = ref(0)
    const isFlipped = ref(false)
    const selectedCategory = ref('all')
    const loading = ref(false)
    const stats = ref({ known: 0, learning: 0, total: 0 })

    // Sample flashcards based on your NewsNeuron codebase
    const sampleCards = [
      {
        id: '1',
        category: 'FastAPI Routes',
        question: 'What are the main API endpoints in NewsNeuron?',
        answer: `<strong>Core Endpoints:</strong><br/>
        • <code>/health</code> - Health check endpoint<br/>
        • <code>/ingest-data</code> - Fetch & store news articles<br/>
        • <code>/chat</code> - Chat with AI agent (streaming/non-streaming)<br/>
        • <code>/search-articles</code> - Search stored articles by similarity`,
        difficulty: 'easy',
        tags: ['fastapi', 'endpoints', 'api'],
        status: 'new'
      },
      {
        id: '2',
        category: 'LangGraph Agent',
        question: 'Which tools are available in the NewsNeuron agent?',
        answer: `<strong>Available Tools:</strong><br/>
        • <code>fetch_gnews</code> - Get latest news articles from API<br/>
        • <code>search_articles</code> - Search stored articles using semantic similarity<br/>
        • <code>query_knowledge_graph</code> - Query Neo4j for entity relationships<br/>
        • <code>search_entity_relationships</code> - Find direct relationships between entities`,
        difficulty: 'medium',
        tags: ['langgraph', 'tools', 'agent'],
        status: 'new'
      },
      {
        id: '3',
        category: 'Data Models',
        question: 'What are the key fields in the NewsArticle Pydantic model?',
        answer: `<strong>NewsArticle Fields:</strong><br/>
        • <code>id, title, content, summary</code> - Basic article info<br/>
        • <code>url, source, author</code> - Source information<br/>
        • <code>published_date</code> - Publication timestamp<br/>
        • <code>categories, entities</code> - Classification data<br/>
        • <code>embedding, relevance_score</code> - ML features`,
        difficulty: 'medium',
        tags: ['pydantic', 'models', 'data'],
        status: 'new'
      },
      {
        id: '4',
        category: 'Vector Store',
        question: 'How does the vector storage system work in NewsNeuron?',
        answer: `<strong>Vector Storage Architecture:</strong><br/>
        • <strong>PostgreSQL</strong> - Stores article metadata and content<br/>
        • <strong>Pinecone</strong> - Stores embeddings for semantic search<br/>
        • <strong>Sentence Transformers</strong> - Generates embeddings (all-MiniLM-L6-v2)<br/>
        • <strong>Hybrid approach</strong> - Combines relational and vector data`,
        difficulty: 'hard',
        tags: ['vector-store', 'pinecone', 'postgresql', 'embeddings'],
        status: 'new'
      },
      {
        id: '5',
        category: 'Configuration',
        question: 'What are the main configuration settings in Settings class?',
        answer: `<strong>Key Settings:</strong><br/>
        • <code>POSTGRES_URL</code> - Database connection<br/>
        • <code>OPENROUTER_API_KEY</code> - AI model access<br/>
        • <code>PINECONE_API_KEY</code> - Vector database<br/>
        • <code>NEO4J_URI/USER/PASSWORD</code> - Knowledge graph<br/>
        • <code>AGENT_MODEL</code> - AI model (gpt-4o-mini)<br/>
        • <code>EMBEDDING_MODEL</code> - Sentence transformer model`,
        difficulty: 'easy',
        tags: ['config', 'environment', 'settings'],
        status: 'new'
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
      updateStats()
      
      // Auto advance to next card
      setTimeout(() => {
        if (currentIndex.value < filteredCards.value.length - 1) {
          nextCard()
        }
      }, 500)
    }

    const updateStats = () => {
      const known = flashcards.value.filter(card => card.status === 'known').length
      const learning = flashcards.value.filter(card => card.status === 'learning').length
      stats.value = {
        known,
        learning,
        total: flashcards.value.length
      }
    }

    const filterCards = () => {
      currentIndex.value = 0
      isFlipped.value = false
    }

    const formatAnswer = (answer) => {
      return answer.replace(/\n/g, '<br/>')
    }

    // API integration
    const generateFromAPI = async () => {
      loading.value = true
      try {
        // Simulate API call to your NewsNeuron backend
        await new Promise(resolve => setTimeout(resolve, 1500))
        
        // In real implementation, you would call:
        // const response = await fetch('/api/generate-flashcards')
        // const newCards = await response.json()
        
        const newCards = [
          {
            id: Date.now().toString(),
            category: 'API Integration',
            question: 'How does the streaming chat work in FastAPI?',
            answer: `<strong>Streaming Implementation:</strong><br/>
            • Uses <code>StreamingResponse</code> with Server-Sent Events<br/>
            • Streams agent responses word by word<br/>
            • Includes tool call information<br/>
            • Handles async graph execution with <code>agent_graph.astream()</code>`,
            difficulty: 'hard',
            tags: ['streaming', 'sse', 'fastapi'],
            status: 'new'
          }
        ]
        
        flashcards.value = [...flashcards.value, ...newCards]
        updateStats()
      } catch (error) {
        console.error('Failed to generate cards:', error)
      } finally {
        loading.value = false
      }
    }

    // Lifecycle
    onMounted(() => {
      flashcards.value = sampleCards
      updateStats()
    })

    return {
      // State
      flashcards,
      currentIndex,
      isFlipped,
      selectedCategory,
      loading,
      stats,
      
      // Computed
      filteredCards,
      categories,
      currentCard,
      progressPercentage,
      
      // Methods
      flipCard,
      nextCard,
      previousCard,
      shuffleCards,
      markCard,
      filterCards,
      formatAnswer,
      generateFromAPI
    }
  }
}
</script>

<style scoped>
.flashcard-app {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  color: #1f2937;
  font-size: 2.5rem;
  margin-bottom: 8px;
}

.header p {
  color: #6b7280;
  font-size: 1.1rem;
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
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  gap: 15px;
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