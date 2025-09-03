<template>
  <div class="min-h-screen bg-dark-background">
    <!-- Header -->
    <div class="bg-dark-card border-b border-dark-border">
      <div class="max-w-7xl mx-auto px-6 py-8">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold text-text-primary mb-2">Flashcards</h1>
            <p class="text-text-secondary">
              AI-generated flashcards from your news analysis
            </p>
          </div>
          <div class="flex items-center space-x-4">
            <button
              @click="generateNewFlashcards"
              :disabled="isGenerating"
              class="btn-neuron"
            >
              <SparklesIcon v-if="!isGenerating" class="w-4 h-4 mr-2" />
              <SynapseLoader v-else class="w-4 h-4 mr-2" />
              {{ isGenerating ? 'Generating...' : 'Generate New' }}
            </button>
            <button
              @click="exportFlashcards"
              :disabled="!hasFlashcards"
              class="btn-secondary"
            >
              <DownloadIcon class="w-4 h-4 mr-2" />
              Export
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="max-w-7xl mx-auto px-6 py-8">
      <!-- Control Panel -->
      <div class="bg-dark-card border border-dark-border rounded-2xl p-6 mb-8">
        <div class="flex items-center justify-between mb-6">
          <div class="flex items-center space-x-6">
            <!-- Search -->
            <div class="relative">
              <SearchIcon class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
              <input
                v-model="searchQuery"
                type="text"
                placeholder="Search flashcards..."
                class="input-neuron pl-10 w-64"
              />
            </div>

            <!-- Category Filter -->
            <select v-model="selectedCategory" class="input-neuron w-48">
              <option value="all">All Categories</option>
              <option v-for="category in categories" :key="category" :value="category">
                {{ category }}
              </option>
            </select>

            <!-- Difficulty Filter -->
            <select v-model="selectedDifficulty" class="input-neuron w-40">
              <option value="all">All Levels</option>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <div class="flex items-center space-x-4">
            <!-- View Mode Toggle -->
            <div class="flex bg-dark-background border border-dark-border rounded-lg p-1">
              <button
                v-for="mode in viewModes"
                :key="mode.id"
                @click="currentView = mode.id"
                :class="[
                  'px-3 py-2 rounded text-sm font-medium transition-all',
                  currentView === mode.id
                    ? 'bg-neuron-glow text-white'
                    : 'text-text-secondary hover:text-text-primary'
                ]"
              >
                <component :is="mode.icon" class="w-4 h-4 mr-1 inline" />
                {{ mode.label }}
              </button>
            </div>
          </div>
        </div>

        <!-- Study Mode Controls -->
        <div v-if="currentView === 'study'" class="flex items-center justify-between border-t border-dark-border pt-6">
          <div class="flex items-center space-x-6">
            <div class="text-text-secondary">
              <span class="text-text-primary font-semibold">{{ currentStudyIndex + 1 }}</span>
              of
              <span class="text-text-primary font-semibold">{{ filteredFlashcards.length }}</span>
              cards
            </div>
            
            <div class="flex items-center space-x-3">
              <span class="text-text-secondary text-sm">Auto-flip:</span>
              <label class="relative inline-flex items-center cursor-pointer">
                <input v-model="autoFlip" type="checkbox" class="sr-only peer" />
                <div class="w-11 h-6 bg-dark-background border border-dark-border peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-neuron-glow"></div>
              </label>
            </div>

            <div v-if="autoFlip" class="flex items-center space-x-2">
              <span class="text-text-secondary text-sm">Timer:</span>
              <select v-model="autoFlipDelay" class="input-neuron w-20 text-sm">
                <option value="3">3s</option>
                <option value="5">5s</option>
                <option value="8">8s</option>
                <option value="10">10s</option>
              </select>
            </div>
          </div>

          <div class="flex items-center space-x-3">
            <button
              @click="previousCard"
              :disabled="currentStudyIndex === 0"
              class="btn-secondary"
            >
              <ChevronLeftIcon class="w-4 h-4" />
            </button>
            <button
              @click="flipCard"
              class="btn-neuron px-6"
            >
              {{ isCardFlipped ? 'Show Question' : 'Show Answer' }}
            </button>
            <button
              @click="nextCard"
              :disabled="currentStudyIndex >= filteredFlashcards.length - 1"
              class="btn-secondary"
            >
              <ChevronRightIcon class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <!-- Main Content -->
      <div class="grid grid-cols-1 xl:grid-cols-4 gap-8">
        <!-- Flashcards Content -->
        <div :class="currentView === 'study' ? 'xl:col-span-4' : 'xl:col-span-3'">
          <!-- Study Mode -->
          <div v-if="currentView === 'study'" class="bg-dark-card border border-dark-border rounded-2xl p-8">
            <div v-if="filteredFlashcards.length > 0" class="text-center">
              <div class="perspective-1000 mb-8">
                <div
                  :class="[
                    'flashcard relative w-full max-w-2xl mx-auto h-96 transition-transform duration-700 preserve-3d cursor-pointer',
                    { 'rotate-y-180': isCardFlipped }
                  ]"
                  @click="flipCard"
                >
                  <!-- Front (Question) -->
                  <div class="flashcard-face flashcard-front absolute inset-0 backface-hidden bg-gradient-to-br from-neuron-glow/10 to-blue-500/10 border-2 border-neuron-glow/30 rounded-2xl p-8">
                    <div class="h-full flex flex-col">
                      <div class="flex justify-between items-start mb-6">
                        <div class="flex items-center space-x-3">
                          <div
                            :class="[
                              'px-3 py-1 rounded-lg text-xs font-medium',
                              getDifficultyClass(currentFlashcard.difficulty)
                            ]"
                          >
                            {{ currentFlashcard.difficulty }}
                          </div>
                          <span class="px-3 py-1 bg-dark-background/50 rounded-lg text-xs text-text-secondary">
                            {{ currentFlashcard.category }}
                          </span>
                        </div>
                        <QuestionMarkCircleIcon class="w-8 h-8 text-neuron-glow" />
                      </div>
                      
                      <div class="flex-1 flex items-center justify-center">
                        <h2 class="text-2xl font-semibold text-text-primary text-center leading-relaxed">
                          {{ currentFlashcard.question }}
                        </h2>
                      </div>

                      <div class="text-center">
                        <p class="text-text-muted text-sm">Click to reveal answer</p>
                      </div>
                    </div>
                  </div>

                  <!-- Back (Answer) -->
                  <div class="flashcard-face flashcard-back absolute inset-0 backface-hidden rotate-y-180 bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-2 border-green-500/30 rounded-2xl p-8">
                    <div class="h-full flex flex-col">
                      <div class="flex justify-between items-start mb-6">
                        <div class="flex items-center space-x-3">
                          <div
                            :class="[
                              'px-3 py-1 rounded-lg text-xs font-medium',
                              getDifficultyClass(currentFlashcard.difficulty)
                            ]"
                          >
                            {{ currentFlashcard.difficulty }}
                          </div>
                          <span class="px-3 py-1 bg-dark-background/50 rounded-lg text-xs text-text-secondary">
                            {{ currentFlashcard.category }}
                          </span>
                        </div>
                        <CheckCircleIcon class="w-8 h-8 text-green-500" />
                      </div>
                      
                      <div class="flex-1 flex items-center justify-center">
                        <div class="text-center">
                          <h2 class="text-xl font-semibold text-text-primary mb-4 leading-relaxed">
                            {{ currentFlashcard.answer }}
                          </h2>
                          <p v-if="currentFlashcard.explanation" class="text-text-secondary text-base">
                            {{ currentFlashcard.explanation }}
                          </p>
                        </div>
                      </div>

                      <div class="flex justify-center space-x-4">
                        <button
                          @click.stop="markCorrect"
                          class="btn-success"
                        >
                          <CheckIcon class="w-4 h-4 mr-2" />
                          Correct
                        </button>
                        <button
                          @click.stop="markIncorrect"
                          class="btn-danger"
                        >
                          <XMarkIcon class="w-4 h-4 mr-2" />
                          Incorrect
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Progress Bar -->
              <div class="w-full bg-dark-background rounded-full h-2 mb-4">
                <div
                  class="bg-neuron-glow h-2 rounded-full transition-all duration-300"
                  :style="{ width: `${studyProgress}%` }"
                ></div>
              </div>
              <p class="text-text-secondary text-sm">
                Progress: {{ Math.round(studyProgress) }}% complete
              </p>
            </div>

            <div v-else class="text-center py-16">
              <AcademicCapIcon class="w-16 h-16 mx-auto mb-4 text-text-muted" />
              <h3 class="text-lg font-medium text-text-secondary mb-2">No flashcards to study</h3>
              <p class="text-text-muted">Generate or create flashcards to start studying.</p>
            </div>
          </div>

          <!-- Grid Mode -->
          <div v-else class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <div class="flex items-center justify-between mb-6">
              <h2 class="text-xl font-semibold text-text-primary">
                Flashcard Collection
                <span v-if="filteredFlashcards.length > 0" class="text-text-secondary text-base font-normal ml-2">
                  ({{ filteredFlashcards.length }} cards)
                </span>
              </h2>
              
              <button
                @click="showCreateModal = true"
                class="btn-neuron"
              >
                <PlusIcon class="w-4 h-4 mr-2" />
                Create Card
              </button>
            </div>

            <!-- Flashcards Grid -->
            <div v-if="filteredFlashcards.length > 0" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div
                v-for="flashcard in paginatedFlashcards"
                :key="flashcard.id"
                class="group relative bg-dark-background border border-dark-border rounded-xl p-6 hover:border-neuron-glow/50 transition-all cursor-pointer"
                @click="viewFlashcard(flashcard)"
              >
                <!-- Category & Difficulty -->
                <div class="flex justify-between items-start mb-4">
                  <span class="px-2 py-1 bg-dark-card rounded-lg text-xs text-text-secondary">
                    {{ flashcard.category }}
                  </span>
                  <div class="flex items-center space-x-2">
                    <div
                      :class="[
                        'px-2 py-1 rounded text-xs font-medium',
                        getDifficultyClass(flashcard.difficulty)
                      ]"
                    >
                      {{ flashcard.difficulty }}
                    </div>
                    <div class="opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        @click.stop="editFlashcard(flashcard)"
                        class="p-1 text-text-muted hover:text-neuron-glow transition-colors"
                      >
                        <PencilIcon class="w-4 h-4" />
                      </button>
                      <button
                        @click.stop="deleteFlashcard(flashcard.id)"
                        class="p-1 text-text-muted hover:text-red-400 transition-colors"
                      >
                        <TrashIcon class="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>

                <!-- Question -->
                <h3 class="text-text-primary font-medium mb-3 line-clamp-3 group-hover:text-neuron-glow transition-colors">
                  {{ flashcard.question }}
                </h3>

                <!-- Stats -->
                <div class="flex items-center justify-between text-sm text-text-muted">
                  <div class="flex items-center space-x-3">
                    <span class="flex items-center">
                      <CheckIcon class="w-3 h-3 mr-1 text-green-500" />
                      {{ flashcard.correctCount || 0 }}
                    </span>
                    <span class="flex items-center">
                      <XMarkIcon class="w-3 h-3 mr-1 text-red-500" />
                      {{ flashcard.incorrectCount || 0 }}
                    </span>
                  </div>
                  <span>{{ formatDate(flashcard.createdAt) }}</span>
                </div>
              </div>
            </div>

            <!-- Empty State -->
            <div v-else class="text-center py-16">
              <AcademicCapIcon class="w-16 h-16 mx-auto mb-4 text-text-muted" />
              <h3 class="text-lg font-medium text-text-secondary mb-2">No flashcards found</h3>
              <p class="text-text-muted mb-6">Create your first flashcard or generate them from articles.</p>
              <button
                @click="generateNewFlashcards"
                class="btn-neuron"
              >
                <SparklesIcon class="w-4 h-4 mr-2" />
                Generate Flashcards
              </button>
            </div>

            <!-- Pagination -->
            <div v-if="totalPages > 1" class="flex justify-center mt-8">
              <div class="flex items-center space-x-2">
                <button
                  @click="currentPage = Math.max(1, currentPage - 1)"
                  :disabled="currentPage === 1"
                  class="btn-secondary"
                >
                  <ChevronLeftIcon class="w-4 h-4" />
                </button>
                
                <span class="px-4 py-2 text-text-secondary">
                  Page {{ currentPage }} of {{ totalPages }}
                </span>
                
                <button
                  @click="currentPage = Math.min(totalPages, currentPage + 1)"
                  :disabled="currentPage === totalPages"
                  class="btn-secondary"
                >
                  <ChevronRightIcon class="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Sidebar (only in grid mode) -->
        <div v-if="currentView === 'grid'" class="space-y-6">
          <!-- Study Statistics -->
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Study Stats</h3>
            <div class="space-y-4">
              <div class="flex justify-between items-center">
                <span class="text-text-secondary">Total Cards</span>
                <span class="text-text-primary font-semibold">{{ flashcards.length }}</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-text-secondary">Studied Today</span>
                <span class="text-text-primary font-semibold">{{ studiedToday }}</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-text-secondary">Accuracy Rate</span>
                <span class="text-text-primary font-semibold">{{ accuracyRate }}%</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-text-secondary">Mastery Level</span>
                <span class="text-text-primary font-semibold">{{ masteryLevel }}%</span>
              </div>
            </div>
          </div>

          <!-- Categories -->
          <div class="bg-dark-card border border-dark-border rounded-2xl p-6">
            <h3 class="text-lg font-semibold text-text-primary mb-4">Categories</h3>
            <div class="space-y-2">
              <div
                v-for="category in categoryStats"
                :key="category.name"
                class="flex justify-between items-center p-2 rounded-lg hover:bg-dark-background/50 transition-colors cursor-pointer"
                @click="selectedCategory = category.name"
              >
                <span class="text-text-secondary">{{ category.name }}</span>
                <span class="text-text-muted text-sm">{{ category.count }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Create/Edit Modal -->
    <div v-if="showCreateModal || editingFlashcard" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div class="bg-dark-card border border-dark-border rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div class="p-6 border-b border-dark-border">
          <h2 class="text-xl font-semibold text-text-primary">
            {{ editingFlashcard ? 'Edit Flashcard' : 'Create New Flashcard' }}
          </h2>
        </div>
        
        <div class="p-6 space-y-6">
          <div>
            <label class="block text-sm font-medium text-text-primary mb-3">Question</label>
            <textarea
              v-model="flashcardForm.question"
              placeholder="Enter your question..."
              class="input-neuron h-24 resize-none"
            ></textarea>
          </div>

          <div>
            <label class="block text-sm font-medium text-text-primary mb-3">Answer</label>
            <textarea
              v-model="flashcardForm.answer"
              placeholder="Enter the answer..."
              class="input-neuron h-24 resize-none"
            ></textarea>
          </div>

          <div>
            <label class="block text-sm font-medium text-text-primary mb-3">Explanation (Optional)</label>
            <textarea
              v-model="flashcardForm.explanation"
              placeholder="Add additional explanation..."
              class="input-neuron h-20 resize-none"
            ></textarea>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">Category</label>
              <input
                v-model="flashcardForm.category"
                type="text"
                placeholder="e.g., Science, History"
                class="input-neuron"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-text-primary mb-3">Difficulty</label>
              <select v-model="flashcardForm.difficulty" class="input-neuron">
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>
          </div>
        </div>

        <div class="p-6 border-t border-dark-border flex justify-end space-x-4">
          <button
            @click="cancelEdit"
            class="btn-secondary"
          >
            Cancel
          </button>
          <button
            @click="saveFlashcard"
            :disabled="!flashcardForm.question.trim() || !flashcardForm.answer.trim()"
            class="btn-neuron"
          >
            {{ editingFlashcard ? 'Update' : 'Create' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  MagnifyingGlassIcon as SearchIcon,
  SparklesIcon,
  ArrowDownTrayIcon as DownloadIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  QuestionMarkCircleIcon,
  CheckCircleIcon,
  CheckIcon,
  XMarkIcon,
  AcademicCapIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
  Squares2X2Icon,
  BookOpenIcon
} from '@heroicons/vue/24/outline'
import SynapseLoader from '@/components/ui/SynapseLoader.vue'

export default {
  name: 'FlashcardsView',
  components: {
    SearchIcon,
    SparklesIcon,
    DownloadIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    QuestionMarkCircleIcon,
    CheckCircleIcon,
    CheckIcon,
    XMarkIcon,
    AcademicCapIcon,
    PlusIcon,
    PencilIcon,
    TrashIcon,
    Squares2X2Icon,
    BookOpenIcon,
    SynapseLoader
  },
  setup() {
    const router = useRouter()

    // State
    const flashcards = ref([])
    const searchQuery = ref('')
    const selectedCategory = ref('all')
    const selectedDifficulty = ref('all')
    const currentView = ref('grid')
    const currentPage = ref(1)
    const cardsPerPage = 12
    const isGenerating = ref(false)

    // Study Mode State
    const currentStudyIndex = ref(0)
    const isCardFlipped = ref(false)
    const autoFlip = ref(false)
    const autoFlipDelay = ref(5)
    let autoFlipTimer = null

    // Modal State
    const showCreateModal = ref(false)
    const editingFlashcard = ref(null)
    const flashcardForm = ref({
      question: '',
      answer: '',
      explanation: '',
      category: '',
      difficulty: 'medium'
    })

    // View Modes
    const viewModes = [
      { id: 'grid', label: 'Browse', icon: Squares2X2Icon },
      { id: 'study', label: 'Study', icon: BookOpenIcon }
    ]

    // Generate sample flashcards
    const generateSampleFlashcards = () => {
      return [
        {
          id: 1,
          question: 'What is the primary cause of the recent surge in renewable energy investments?',
          answer: 'Government subsidies and decreasing technology costs',
          explanation: 'Multiple factors including policy support, technological advances, and environmental concerns have driven this trend.',
          category: 'Technology',
          difficulty: 'medium',
          correctCount: 8,
          incorrectCount: 2,
          createdAt: '2024-01-15T10:30:00Z'
        },
        {
          id: 2,
          question: 'Which cryptocurrency regulatory framework was recently approved by the EU?',
          answer: 'Markets in Crypto-Assets (MiCA) regulation',
          explanation: 'This comprehensive framework aims to provide legal clarity for crypto assets across European Union member states.',
          category: 'Finance',
          difficulty: 'hard',
          correctCount: 5,
          incorrectCount: 7,
          createdAt: '2024-01-14T15:45:00Z'
        },
        {
          id: 3,
          question: 'What was the main outcome of the recent climate summit?',
          answer: 'Agreement on fossil fuel transition timeline',
          explanation: 'Countries committed to specific targets for reducing fossil fuel dependency by 2030.',
          category: 'Environment',
          difficulty: 'easy',
          correctCount: 12,
          incorrectCount: 1,
          createdAt: '2024-01-13T09:15:00Z'
        },
        {
          id: 4,
          question: 'How has AI technology impacted healthcare diagnosis accuracy?',
          answer: 'Improved accuracy by 25-40% in most specialties',
          explanation: 'Machine learning algorithms have shown significant improvements in medical imaging and diagnostic processes.',
          category: 'Healthcare',
          difficulty: 'medium',
          correctCount: 9,
          incorrectCount: 3,
          createdAt: '2024-01-12T14:20:00Z'
        },
        {
          id: 5,
          question: 'What is the significance of the recent space mission to Europa?',
          answer: 'First mission to search for signs of life on Jupiter\'s moon',
          explanation: 'The mission aims to analyze the subsurface ocean beneath Europa\'s ice shell for potential biosignatures.',
          category: 'Science',
          difficulty: 'hard',
          correctCount: 4,
          incorrectCount: 8,
          createdAt: '2024-01-11T11:00:00Z'
        },
        {
          id: 6,
          question: 'Which company announced the largest acquisition in tech history?',
          answer: 'Microsoft\'s acquisition of Activision Blizzard',
          explanation: 'The $68.7 billion deal was completed after regulatory approval, marking the largest gaming industry acquisition.',
          category: 'Business',
          difficulty: 'easy',
          correctCount: 15,
          incorrectCount: 2,
          createdAt: '2024-01-10T16:30:00Z'
        }
      ]
    }

    // Computed Properties
    const hasFlashcards = computed(() => flashcards.value.length > 0)

    const categories = computed(() => {
      const cats = [...new Set(flashcards.value.map(card => card.category))]
      return cats.sort()
    })

    const filteredFlashcards = computed(() => {
      return flashcards.value.filter(card => {
        const matchesSearch = searchQuery.value === '' || 
          card.question.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
          card.answer.toLowerCase().includes(searchQuery.value.toLowerCase())
        
        const matchesCategory = selectedCategory.value === 'all' || 
          card.category === selectedCategory.value
        
        const matchesDifficulty = selectedDifficulty.value === 'all' || 
          card.difficulty === selectedDifficulty.value

        return matchesSearch && matchesCategory && matchesDifficulty
      })
    })

    const totalPages = computed(() => Math.ceil(filteredFlashcards.value.length / cardsPerPage))

    const paginatedFlashcards = computed(() => {
      const start = (currentPage.value - 1) * cardsPerPage
      const end = start + cardsPerPage
      return filteredFlashcards.value.slice(start, end)
    })

    const currentFlashcard = computed(() => {
      return filteredFlashcards.value[currentStudyIndex.value] || {}
    })

    const studyProgress = computed(() => {
      if (filteredFlashcards.value.length === 0) return 0
      return ((currentStudyIndex.value + 1) / filteredFlashcards.value.length) * 100
    })

    const studiedToday = computed(() => {
      // Mock calculation - in real app would track actual study sessions
      return Math.floor(Math.random() * 20) + 5
    })

    const accuracyRate = computed(() => {
      if (!hasFlashcards.value) return 0
      const totalCorrect = flashcards.value.reduce((sum, card) => sum + (card.correctCount || 0), 0)
      const totalIncorrect = flashcards.value.reduce((sum, card) => sum + (card.incorrectCount || 0), 0)
      const total = totalCorrect + totalIncorrect
      return total > 0 ? Math.round((totalCorrect / total) * 100) : 0
    })

    const masteryLevel = computed(() => {
      if (!hasFlashcards.value) return 0
      const masteredCards = flashcards.value.filter(card => 
        (card.correctCount || 0) >= 3 && (card.correctCount || 0) > (card.incorrectCount || 0) * 2
      ).length
      return Math.round((masteredCards / flashcards.value.length) * 100)
    })

    const categoryStats = computed(() => {
      const stats = {}
      flashcards.value.forEach(card => {
        stats[card.category] = (stats[card.category] || 0) + 1
      })
      return Object.entries(stats)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
    })

    // Methods
    const generateNewFlashcards = async () => {
      isGenerating.value = true
      
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        const newFlashcards = [
          {
            id: Date.now(),
            question: 'What is the latest development in quantum computing breakthrough?',
            answer: 'IBM achieved quantum advantage in optimization problems',
            explanation: 'Recent advances show quantum computers can solve certain optimization problems faster than classical computers.',
            category: 'Technology',
            difficulty: 'hard',
            correctCount: 0,
            incorrectCount: 0,
            createdAt: new Date().toISOString()
          },
          {
            id: Date.now() + 1,
            question: 'Which country announced the world\'s largest offshore wind farm project?',
            answer: 'United Kingdom',
            explanation: 'The UK announced a massive offshore wind project aimed at achieving net-zero carbon emissions by 2050.',
            category: 'Environment',
            difficulty: 'medium',
            correctCount: 0,
            incorrectCount: 0,
            createdAt: new Date().toISOString()
          }
        ]
        
        flashcards.value.push(...newFlashcards)
      } finally {
        isGenerating.value = false
      }
    }

    const exportFlashcards = () => {
      const data = flashcards.value.map(card => ({
        question: card.question,
        answer: card.answer,
        explanation: card.explanation,
        category: card.category,
        difficulty: card.difficulty
      }))
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `flashcards-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
    }

    const flipCard = () => {
      isCardFlipped.value = !isCardFlipped.value
      
      if (autoFlip.value && !isCardFlipped.value) {
        startAutoFlipTimer()
      }
    }

    const startAutoFlipTimer = () => {
      clearTimeout(autoFlipTimer)
      autoFlipTimer = setTimeout(() => {
        if (!isCardFlipped.value) {
          flipCard()
        }
      }, autoFlipDelay.value * 1000)
    }

    const previousCard = () => {
      if (currentStudyIndex.value > 0) {
        currentStudyIndex.value--
        isCardFlipped.value = false
        if (autoFlip.value) startAutoFlipTimer()
      }
    }

    const nextCard = () => {
      if (currentStudyIndex.value < filteredFlashcards.value.length - 1) {
        currentStudyIndex.value++
        isCardFlipped.value = false
        if (autoFlip.value) startAutoFlipTimer()
      }
    }

    const markCorrect = () => {
      const card = currentFlashcard.value
      if (card.id) {
        const index = flashcards.value.findIndex(f => f.id === card.id)
        if (index >= 0) {
          flashcards.value[index].correctCount = (flashcards.value[index].correctCount || 0) + 1
        }
        nextCard()
      }
    }

    const markIncorrect = () => {
      const card = currentFlashcard.value
      if (card.id) {
        const index = flashcards.value.findIndex(f => f.id === card.id)
        if (index >= 0) {
          flashcards.value[index].incorrectCount = (flashcards.value[index].incorrectCount || 0) + 1
        }
        nextCard()
      }
    }

    const viewFlashcard = (flashcard) => {
      const index = filteredFlashcards.value.findIndex(f => f.id === flashcard.id)
      if (index >= 0) {
        currentStudyIndex.value = index
        currentView.value = 'study'
        isCardFlipped.value = false
        if (autoFlip.value) startAutoFlipTimer()
      }
    }

    const editFlashcard = (flashcard) => {
      editingFlashcard.value = flashcard
      flashcardForm.value = { ...flashcard }
    }

    const deleteFlashcard = (id) => {
      if (confirm('Are you sure you want to delete this flashcard?')) {
        const index = flashcards.value.findIndex(f => f.id === id)
        if (index >= 0) {
          flashcards.value.splice(index, 1)
        }
      }
    }

    const saveFlashcard = () => {
      if (!flashcardForm.value.question.trim() || !flashcardForm.value.answer.trim()) return

      if (editingFlashcard.value) {
        // Update existing
        const index = flashcards.value.findIndex(f => f.id === editingFlashcard.value.id)
        if (index >= 0) {
          flashcards.value[index] = {
            ...flashcards.value[index],
            ...flashcardForm.value
          }
        }
      } else {
        // Create new
        const newFlashcard = {
          id: Date.now(),
          ...flashcardForm.value,
          correctCount: 0,
          incorrectCount: 0,
          createdAt: new Date().toISOString()
        }
        flashcards.value.push(newFlashcard)
      }

      cancelEdit()
    }

    const cancelEdit = () => {
      showCreateModal.value = false
      editingFlashcard.value = null
      flashcardForm.value = {
        question: '',
        answer: '',
        explanation: '',
        category: '',
        difficulty: 'medium'
      }
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

    const getDifficultyClass = (difficulty) => {
      switch (difficulty) {
        case 'easy': return 'bg-green-500/20 text-green-400 border border-green-500/30'
        case 'medium': return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
        case 'hard': return 'bg-red-500/20 text-red-400 border border-red-500/30'
        default: return 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
      }
    }

    // Watchers
    watch(autoFlip, (newValue) => {
      if (newValue && currentView.value === 'study' && !isCardFlipped.value) {
        startAutoFlipTimer()
      } else {
        clearTimeout(autoFlipTimer)
      }
    })

    watch(autoFlipDelay, () => {
      if (autoFlip.value && currentView.value === 'study' && !isCardFlipped.value) {
        startAutoFlipTimer()
      }
    })

    // Lifecycle
    onMounted(() => {
      flashcards.value = generateSampleFlashcards()
    })

    onUnmounted(() => {
      clearTimeout(autoFlipTimer)
    })

    return {
      // State
      flashcards,
      searchQuery,
      selectedCategory,
      selectedDifficulty,
      currentView,
      currentPage,
      isGenerating,
      currentStudyIndex,
      isCardFlipped,
      autoFlip,
      autoFlipDelay,
      showCreateModal,
      editingFlashcard,
      flashcardForm,
      viewModes,

      // Computed
      hasFlashcards,
      categories,
      filteredFlashcards,
      totalPages,
      paginatedFlashcards,
      currentFlashcard,
      studyProgress,
      studiedToday,
      accuracyRate,
      masteryLevel,
      categoryStats,

      // Methods
      generateNewFlashcards,
      exportFlashcards,
      flipCard,
      previousCard,
      nextCard,
      markCorrect,
      markIncorrect,
      viewFlashcard,
      editFlashcard,
      deleteFlashcard,
      saveFlashcard,
      cancelEdit,
      formatDate,
      getDifficultyClass
    }
  }
}
</script>

<style scoped>
.perspective-1000 {
  perspective: 1000px;
}

.preserve-3d {
  transform-style: preserve-3d;
}

.backface-hidden {
  backface-visibility: hidden;
}

.rotate-y-180 {
  transform: rotateY(180deg);
}

.flashcard-face {
  transform-style: preserve-3d;
}

.flashcard-back {
  transform: rotateY(180deg);
}

.line-clamp-3 {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.btn-success {
  @apply px-4 py-2 bg-green-500/20 text-green-400 border border-green-500/30 rounded-lg hover:bg-green-500/30 transition-colors font-medium;
}

.btn-danger {
  @apply px-4 py-2 bg-red-500/20 text-red-400 border border-red-500/30 rounded-lg hover:bg-red-500/30 transition-colors font-medium;
}
</style>
