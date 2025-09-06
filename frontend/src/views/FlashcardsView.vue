<template>
  <div class="flashcard-view p-4 sm:p-6 lg:p-8 bg-neuron-bg-primary min-h-full">
    <div class="max-w-7xl mx-auto">
      
      <!-- Header -->
      <header class="text-center mb-8">
        <h1 class="text-4xl font-bold text-neuron-text-primary tracking-tight">NewsNeuron Flashcards</h1>
        <p class="mt-2 text-lg text-neuron-text-secondary">Your daily dose of AI-powered news knowledge.</p>
      </header>

      <!-- Stats -->
      <section class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div class="bg-neuron-bg-content p-4 rounded-lg shadow-md flex items-center">
          <div class="text-3xl mr-4">📚</div>
          <div>
            <div class="text-2xl font-bold text-neuron-text-primary">{{ stats.total }}</div>
            <div class="text-sm text-neuron-text-secondary">Total Cards</div>
          </div>
        </div>
        <div class="bg-neuron-bg-content p-4 rounded-lg shadow-md flex items-center">
          <div class="text-3xl mr-4">✅</div>
          <div>
            <div class="text-2xl font-bold text-green-400">{{ stats.known }}</div>
            <div class="text-sm text-neuron-text-secondary">Mastered</div>
          </div>
        </div>
        <div class="bg-neuron-bg-content p-4 rounded-lg shadow-md flex items-center">
          <div class="text-3xl mr-4">🎯</div>
          <div>
            <div class="text-2xl font-bold text-yellow-400">{{ stats.learning }}</div>
            <div class="text-sm text-neuron-text-secondary">Learning</div>
          </div>
        </div>
        <div class="bg-neuron-bg-content p-4 rounded-lg shadow-md flex items-center">
          <div class="text-3xl mr-4">⏰</div>
          <div>
            <div class="text-2xl font-bold text-blue-400">{{ stats.dueToday }}</div>
            <div class="text-sm text-neuron-text-secondary">Due Today</div>
          </div>
        </div>
      </section>

      <!-- Controls -->
      <section class="mb-8 p-4 bg-neuron-bg-content rounded-lg shadow-md">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label for="topic-select" class="block text-sm font-medium text-neuron-text-secondary mb-1">Topic</label>
            <select id="topic-select" v-model="selectedTopic" class="input-neuron w-full">
              <option v-for="topic in newsTopics" :key="topic" :value="topic">{{ topic }}</option>
            </select>
          </div>
          <div>
            <label for="category-select" class="block text-sm font-medium text-neuron-text-secondary mb-1">Category</label>
            <select id="category-select" v-model="selectedCategory" @change="filterCards" class="input-neuron w-full">
              <option value="all">All Categories</option>
              <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
            </select>
          </div>
          <div class="flex items-end">
            <button @click="generateNewsCards" class="btn-neuron w-full" :disabled="loading">
              <span v-if="loading">Generating...</span>
              <span v-else>Generate Cards</span>
            </button>
          </div>
        </div>
      </section>

      <!-- Flashcard -->
      <section v-if="filteredCards.length > 0" class="relative">
        <div class="w-full h-2 bg-neuron-bg-content rounded-full mb-2">
          <div class="h-2 bg-neuron-glow rounded-full" :style="{ width: progressPercentage + '%' }"></div>
        </div>
        <div class="text-right text-sm text-neuron-text-secondary mb-4">{{ currentIndex + 1 }} / {{ filteredCards.length }}</div>
        
        <div class="relative h-96 perspective-1000">
          <div class="absolute w-full h-full transition-transform duration-700 transform-style-3d" :class="{'rotate-y-180': isFlipped}">
            <!-- Front -->
            <div class="absolute w-full h-full backface-hidden bg-neuron-bg-content rounded-xl shadow-lg p-6 flex flex-col" @click="flipCard">
              <div class="flex-grow">
                <div class="flex justify-between items-start mb-4">
                  <span class="px-3 py-1 text-sm rounded-full bg-blue-500/20 text-blue-300">{{ currentCard.category }}</span>
                  <span class="px-3 py-1 text-sm rounded-full" :class="difficultyClass(currentCard.difficulty)">{{ currentCard.difficulty }}</span>
                </div>
                <h2 class="text-2xl font-semibold text-neuron-text-primary mb-4">{{ currentCard.question }}</h2>
                <p v-if="currentCard.context" class="text-neuron-text-secondary">{{ currentCard.context }}</p>
              </div>
              <div class="text-center text-neuron-text-secondary text-sm">Click to flip</div>
            </div>
            <!-- Back -->
            <div class="absolute w-full h-full backface-hidden bg-neuron-bg-content rounded-xl shadow-lg p-6 flex flex-col rotate-y-180" @click="flipCard">
              <div class="flex-grow">
                <h3 class="text-xl font-semibold text-neuron-text-primary mb-4">Answer</h3>
                <div class="text-neuron-text-secondary space-y-2" v-html="formatAnswer(currentCard.answer)"></div>
                <div v-if="currentCard.source" class="mt-4 text-sm text-neuron-text-secondary">
                  <strong>Source:</strong> {{ currentCard.source }}
                </div>
                <a v-if="currentCard.readMore" :href="currentCard.readMore" target="_blank" class="mt-2 text-blue-400 hover:underline">Read more</a>
              </div>
              <div class="flex justify-around mt-4">
                <button @click.stop="markCard('known')" class="btn-ghost text-green-400 hover:bg-green-500/20">Got it!</button>
                <button @click.stop="markCard('learning')" class="btn-ghost text-yellow-400 hover:bg-yellow-500/20">Learning</button>
                <button @click.stop="markCard('difficult')" class="btn-ghost text-red-400 hover:bg-red-500/20">Difficult</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Navigation -->
        <div class="flex justify-between mt-6">
          <button @click="previousCard" :disabled="currentIndex === 0" class="btn-ghost">Previous</button>
          <button @click="nextCard" :disabled="currentIndex === filteredCards.length - 1" class="btn-ghost">Next</button>
        </div>
      </section>

      <!-- Empty State -->
      <section v-else class="text-center py-16">
        <div class="text-6xl mb-4">🤔</div>
        <h2 class="text-2xl font-semibold text-neuron-text-primary">No flashcards to show</h2>
        <p class="text-neuron-text-secondary mt-2">Generate some cards to get started!</p>
      </section>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';
import { useFlashcardsStore } from '@/stores/flashcards';

const flashcardStore = useFlashcardsStore();

const isFlipped = ref(false);
const selectedCategory = ref('all');
const selectedTopic = ref('technology');
const loading = ref(false);

const newsTopics = ref([
  'politics', 'technology', 'climate change', 'economy', 
  'health', 'sports', 'world news', 'business', 'science'
]);

const stats = computed(() => ({
  total: flashcardStore.flashcards.length,
  known: flashcardStore.flashcards.filter(c => c.status === 'known').length,
  learning: flashcardStore.flashcards.filter(c => c.status === 'learning').length,
  dueToday: flashcardStore.flashcards.filter(c => c.status !== 'known').length, // Simplified
}));

const filteredCards = computed(() => {
  if (selectedCategory.value === 'all') {
    return flashcardStore.flashcards;
  }
  return flashcardStore.flashcards.filter(card => card.category === selectedCategory.value);
});

const categories = computed(() => {
  const cats = [...new Set(flashcardStore.flashcards.map(card => card.category))];
  return cats.sort();
});

const currentIndex = ref(0);

const currentCard = computed(() => {
  return filteredCards.value[currentIndex.value] || {};
});

const progressPercentage = computed(() => {
  if (filteredCards.value.length === 0) return 0;
  return ((currentIndex.value + 1) / filteredCards.value.length) * 100;
});

function flipCard() {
  isFlipped.value = !isFlipped.value;
}

function nextCard() {
  if (currentIndex.value < filteredCards.value.length - 1) {
    currentIndex.value++;
    isFlipped.value = false;
  }
}

function previousCard() {
  if (currentIndex.value > 0) {
    currentIndex.value--;
    isFlipped.value = false;
  }
}

function markCard(status) {
  flashcardStore.updateCardStatus(currentCard.value.id, status);
  if (settings.value.autoAdvance) {
    setTimeout(() => {
      if (currentIndex.value < filteredCards.value.length - 1) {
        nextCard();
      }
    }, 500);
  }
}

async function generateNewsCards() {
  loading.value = true;
  try {
    await flashcardStore.generateFlashcards({ topic: selectedTopic.value });
  } catch (error) {
    console.error('Failed to generate flashcards:', error);
    // You might want to show a toast notification here
  } finally {
    loading.value = false;
  }
}

function filterCards() {
  currentIndex.value = 0;
  isFlipped.value = false;
}

function formatAnswer(answer) {
  return answer.replace(/\n/g, '<br/>');
}

function difficultyClass(difficulty) {
  return {
    'easy': 'bg-green-500/20 text-green-300',
    'medium': 'bg-yellow-500/20 text-yellow-300',
    'hard': 'bg-red-500/20 text-red-300',
  }[difficulty];
}

const settings = ref({
  autoAdvance: true,
});

onMounted(() => {
  if (flashcardStore.flashcards.length === 0) {
    generateNewsCards();
  }
});

watch(selectedTopic, () => {
  generateNewsCards();
});

</script>

<style scoped>
.perspective-1000 {
  perspective: 1000px;
}
.transform-style-3d {
  transform-style: preserve-3d;
}
.rotate-y-180 {
  transform: rotateY(180deg);
}
.backface-hidden {
  backface-visibility: hidden;
}
</style>