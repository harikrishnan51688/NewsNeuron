import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useFlashcardsStore = defineStore('flashcards', () => {
  const flashcards = ref([]);

  async function generateFlashcards({ topic, sourceType = 'recent_news', difficulty = 'medium', count = 10 }) {
    try {
      const response = await fetch('http://localhost:8000/flashcards/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic,
          source_type: sourceType,
          difficulty,
          count,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate flashcards');
      }

      const data = await response.json();

      if (data.success) {
        flashcards.value = data.flashcards;
      } else {
        throw new Error(data.error || 'Generation failed');
      }
    } catch (error) {
      console.error('Failed to generate flashcards:', error);
      // Optionally, you can re-throw the error to be caught in the component
      throw error;
    }
  }

  function updateCardStatus(cardId, status) {
    const card = flashcards.value.find(c => c.id === cardId);
    if (card) {
      card.status = status;
    }
  }

  return { flashcards, generateFlashcards, updateCardStatus };
});