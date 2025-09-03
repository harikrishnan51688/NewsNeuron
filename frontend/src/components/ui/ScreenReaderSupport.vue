<template>
  <div class="sr-only">
    <!-- Screen reader announcements -->
    <div 
      id="sr-announcement" 
      aria-live="polite" 
      aria-atomic="true"
      class="sr-only"
    >
      {{ announcement }}
    </div>
    
    <!-- Skip navigation link -->
    <a 
      href="#main-content" 
      class="skip-link sr-only focus:not-sr-only focus:absolute focus:top-0 focus:left-0 z-50 bg-neuron-bg-primary text-neuron-text-primary px-4 py-2 rounded-md border border-neuron-border"
    >
      Skip to main content
    </a>
    
    <!-- Application description for screen readers -->
    <div id="app-description" class="sr-only">
      NewsNeuron: AI-powered news analysis system that processes news like interconnected neurons, 
      understanding both content and relationships between entities. Navigate using tab key or arrow keys.
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'

// Reactive state
const announcement = ref('')

// Store
const appStore = useAppStore()

// Methods
const announce = (message) => {
  announcement.value = message
  // Clear after announcement to allow repeat announcements
  setTimeout(() => {
    announcement.value = ''
  }, 1000)
}

// Listen for route changes and announce them
onMounted(() => {
  // Initial announcement
  announce('NewsNeuron application loaded. Navigate using tab key or arrow keys.')
  
  // Listen for store announcements
  appStore.$onAction(({ name, after }) => {
    after(() => {
      switch (name) {
        case 'setLoading':
          if (appStore.isLoading) {
            announce('Loading content...')
          } else {
            announce('Content loaded')
          }
          break
        case 'setError':
          if (appStore.error) {
            announce(`Error: ${appStore.error}`)
          }
          break
      }
    })
  })
})

// Expose announce method for external use
defineExpose({
  announce
})
</script>

<style scoped>
/* Screen reader only utility class */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.sr-only:focus,
.focus\:not-sr-only:focus {
  position: static;
  width: auto;
  height: auto;
  padding: inherit;
  margin: inherit;
  overflow: visible;
  clip: auto;
  white-space: normal;
}

/* Skip link styling */
.skip-link {
  transition: all 0.2s ease-in-out;
  text-decoration: none;
}

.skip-link:hover {
  background-color: var(--neuron-glow);
  color: white;
}

.skip-link:focus {
  outline: 2px solid var(--neuron-glow);
  outline-offset: 2px;
}
</style>
