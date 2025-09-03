import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { flashcardsAPI } from "@/services/api";

export const useFlashcardsStore = defineStore("flashcards", () => {
  // State
  const flashcards = ref([]);
  const currentFlashcard = ref(null);
  const isLoading = ref(false);
  const error = ref(null);
  const filters = ref({
    topics: [],
    dateRange: null,
    limit: 10,
  });
  const trendingTopics = ref([]);

  // Getters
  const flashcardCount = computed(() => flashcards.value.length);
  const hasFlashcards = computed(() => flashcards.value.length > 0);
  const filteredFlashcards = computed(() => {
    let filtered = flashcards.value;

    // Apply topic filter
    if (filters.value.topics.length > 0) {
      filtered = filtered.filter((flashcard) =>
        filters.value.topics.some(
          (topic) =>
            flashcard.category?.toLowerCase().includes(topic.toLowerCase()) ||
            flashcard.title.toLowerCase().includes(topic.toLowerCase()),
        ),
      );
    }

    return filtered;
  });

  // Actions
  async function generateFlashcards(requestData = {}) {
    try {
      isLoading.value = true;
      error.value = null;

      const data = {
        topics: filters.value.topics,
        date_range: filters.value.dateRange,
        limit: filters.value.limit,
        ...requestData,
      };

      const response = await flashcardsAPI.generateFlashcards(data);
      flashcards.value = response.data.flashcards;

      return response.data;
    } catch (err) {
      error.value = err.message || "Failed to generate flashcards";
      throw err;
    } finally {
      isLoading.value = false;
    }
  }

  async function getFlashcards(params = {}) {
    try {
      isLoading.value = true;
      error.value = null;

      const queryParams = {
        limit: filters.value.limit,
        topics: filters.value.topics,
        days_back: 7,
        ...params,
      };

      const response = await flashcardsAPI.getFlashcards(queryParams);
      flashcards.value = response.data.flashcards;

      return response.data;
    } catch (err) {
      error.value = err.message || "Failed to fetch flashcards";
      throw err;
    } finally {
      isLoading.value = false;
    }
  }

  async function getRecentFlashcards(limit = 5) {
    try {
      const params = {
        limit,
        days_back: 7,
      };

      const response = await flashcardsAPI.getFlashcards(params);
      return response.data.flashcards;
    } catch (err) {
      console.error("Failed to fetch recent flashcards:", err);
      return [];
    }
  }

  async function getFlashcardDetails(flashcardId) {
    try {
      isLoading.value = true;
      error.value = null;

      const response = await flashcardsAPI.getFlashcardDetails(flashcardId);
      currentFlashcard.value = response.data.flashcard;

      return response.data.flashcard;
    } catch (err) {
      error.value = err.message || "Failed to fetch flashcard details";
      throw err;
    } finally {
      isLoading.value = false;
    }
  }

  async function getTrendingTopics(params = {}) {
    try {
      const queryParams = {
        limit: 10,
        days_back: 7,
        ...params,
      };

      const response = await flashcardsAPI.getTrendingTopics(queryParams);
      trendingTopics.value = response.data.trending_topics;

      return response.data.trending_topics;
    } catch (err) {
      console.error("Failed to fetch trending topics:", err);
      return [];
    }
  }

  function setFilters(newFilters) {
    filters.value = {
      ...filters.value,
      ...newFilters,
    };
  }

  function addTopicFilter(topic) {
    if (!filters.value.topics.includes(topic)) {
      filters.value.topics.push(topic);
    }
  }

  function removeTopicFilter(topic) {
    const index = filters.value.topics.indexOf(topic);
    if (index > -1) {
      filters.value.topics.splice(index, 1);
    }
  }

  function clearTopicFilters() {
    filters.value.topics = [];
  }

  function setDateRange(startDate, endDate) {
    filters.value.dateRange = {
      start_date: startDate,
      end_date: endDate,
    };
  }

  function clearDateRange() {
    filters.value.dateRange = null;
  }

  function clearAllFilters() {
    filters.value = {
      topics: [],
      dateRange: null,
      limit: 10,
    };
  }

  function setCurrentFlashcard(flashcard) {
    currentFlashcard.value = flashcard;
  }

  function clearCurrentFlashcard() {
    currentFlashcard.value = null;
  }

  function addFlashcard(flashcard) {
    flashcards.value.unshift(flashcard);
  }

  function updateFlashcard(flashcardId, updates) {
    const index = flashcards.value.findIndex((f) => f.id === flashcardId);
    if (index !== -1) {
      flashcards.value[index] = {
        ...flashcards.value[index],
        ...updates,
      };
    }
  }

  function removeFlashcard(flashcardId) {
    const index = flashcards.value.findIndex((f) => f.id === flashcardId);
    if (index !== -1) {
      flashcards.value.splice(index, 1);
    }
  }

  function clearFlashcards() {
    flashcards.value = [];
  }

  function clearError() {
    error.value = null;
  }

  // Helper functions
  function getFlashcardsByCategory(category) {
    return flashcards.value.filter(
      (f) => f.category?.toLowerCase() === category.toLowerCase(),
    );
  }

  function getFlashcardsByTopic(topic) {
    return flashcards.value.filter(
      (f) =>
        f.title.toLowerCase().includes(topic.toLowerCase()) ||
        f.summary.toLowerCase().includes(topic.toLowerCase()) ||
        f.key_points.some((point) =>
          point.toLowerCase().includes(topic.toLowerCase()),
        ),
    );
  }

  function searchFlashcards(query) {
    const searchTerm = query.toLowerCase();
    return flashcards.value.filter(
      (f) =>
        f.title.toLowerCase().includes(searchTerm) ||
        f.summary.toLowerCase().includes(searchTerm) ||
        f.key_points.some((point) =>
          point.toLowerCase().includes(searchTerm),
        ) ||
        f.entities.some((entity) =>
          entity.name.toLowerCase().includes(searchTerm),
        ),
    );
  }

  return {
    // State
    flashcards,
    currentFlashcard,
    isLoading,
    error,
    filters,
    trendingTopics,

    // Getters
    flashcardCount,
    hasFlashcards,
    filteredFlashcards,

    // Actions
    generateFlashcards,
    getFlashcards,
    getRecentFlashcards,
    getFlashcardDetails,
    getTrendingTopics,
    setFilters,
    addTopicFilter,
    removeTopicFilter,
    clearTopicFilters,
    setDateRange,
    clearDateRange,
    clearAllFilters,
    setCurrentFlashcard,
    clearCurrentFlashcard,
    addFlashcard,
    updateFlashcard,
    removeFlashcard,
    clearFlashcards,
    clearError,

    // Helper functions
    getFlashcardsByCategory,
    getFlashcardsByTopic,
    searchFlashcards,
  };
});
