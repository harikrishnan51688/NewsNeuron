<template>
  <nav class="navbar bg-gradient-to-r from-neuron-bg-content via-slate-800/60 to-neuron-bg-content border-b border-neuron-border/30 sticky top-0 z-50 backdrop-blur-md">
    <div class="navbar-container max-w-8xl mx-auto px-6 py-2">
      <div class="flex items-center justify-between">
        
        <!-- Left: Logo & Brand -->
        <div class="flex items-center space-x-6">
          <router-link 
            to="/" 
            class="logo-brand group flex items-center space-x-3 hover:scale-105 transition-all duration-300 cursor-pointer rounded-lg p-2 hover:bg-neuron-bg-primary/30"
          >
            <NeuronLogo 
              class="w-10 h-10 text-logo-primary group-hover:text-logo-secondary transition-all duration-300" 
              :animated="true"
              :primaryColor="'#8B5CF6'"
              :secondaryColor="'#A855F7'"
            />
            <div class="flex flex-col">
              <h1 class="text-lg font-display font-semibold logo-text-gradient group-hover:scale-105 transition-all duration-300">
                NewsNeuron
              </h1>
              <p class="text-xs font-sans text-neuron-text-secondary/80 group-hover:text-logo-primary transition-colors duration-300">
                AI News Analysis
              </p>
            </div>
          </router-link>
          
          <!-- Desktop Navigation Links -->
          <div class="hidden lg:flex items-center space-x-1">
            <NavLink
              to="/"
              :icon="LayoutDashboard"
              horizontal
            >
              Dashboard
            </NavLink>
            
            <NavLink
              to="/chat"
              :icon="MessageSquare"
              horizontal
            >
              Chat
            </NavLink>
            
            <NavLink
              to="/flashcards"
              :icon="BookOpen"
              horizontal
              :badge="flashcardCount > 0 ? flashcardCount.toString() : null"
            >
              Flashcards
            </NavLink>
            
            <NavLink
              to="/timeline"
              :icon="Clock"
              horizontal
            >
              Timeline
            </NavLink>
          </div>
        </div>
        
        <!-- Right: Search Only -->
        <div class="flex items-center space-x-4">
          
          <!-- Search Button -->
          <NavLink
            to="/search"
            :icon="Search"
            horizontal
            class="bg-neuron-bg-content/50 border border-neuron-border/50 hover:border-neuron-glow/30 px-5 py-2.5 rounded-lg text-base"
          >
            <span class="hidden md:inline">Search</span>
          </NavLink>
          
          <!-- Mobile Menu Toggle -->
          <button 
            @click="showMobileMenu = !showMobileMenu"
            class="lg:hidden btn-icon p-2"
            :aria-label="showMobileMenu ? 'Close menu' : 'Open menu'"
          >
            <Menu v-if="!showMobileMenu" class="w-5 h-5" />
            <X v-else class="w-5 h-5" />
          </button>
        </div>
      </div>
      
      <!-- Mobile Menu -->
      <div 
        v-if="showMobileMenu"
        class="lg:hidden mt-4 pt-4 border-t border-neuron-border/30"
      >
        <div class="grid grid-cols-2 gap-2">
          <NavLink
            to="/"
            :icon="LayoutDashboard"
            mobile
            @click="showMobileMenu = false"
          >
            Dashboard
          </NavLink>
          
          <NavLink
            to="/chat"
            :icon="MessageSquare"
            mobile
            @click="showMobileMenu = false"
          >
            Chat
          </NavLink>
          
          <NavLink
            to="/search"
            :icon="Search"
            mobile
            @click="showMobileMenu = false"
          >
            Search
          </NavLink>
          
          <NavLink
            to="/timeline"
            :icon="Clock"
            mobile
            @click="showMobileMenu = false"
          >
            Timeline
          </NavLink>
          
          <NavLink
            to="/flashcards"
            :icon="BookOpen"
            mobile
            :badge="flashcardCount > 0 ? flashcardCount.toString() : null"
            @click="showMobileMenu = false"
          >
            Flashcards
          </NavLink>
        </div>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useFlashcardsStore } from '@/stores/flashcards'

// Icons
import {
  LayoutDashboard,
  MessageSquare,
  Search,
  Clock,
  BookOpen,
  Menu,
  X
} from 'lucide-vue-next'

// Components
import NavLink from './NavLink.vue'
import NeuronLogo from '@/components/ui/NeuronLogo.vue'

// State
const showMobileMenu = ref(false)

// Store
const flashcardsStore = useFlashcardsStore()

// Computed
const flashcardCount = computed(() => flashcardsStore.flashcards.length)
</script>

<style scoped>
/* Navbar Styles */
.navbar {
  background: linear-gradient(90deg, 
    rgba(22, 27, 34, 0.95) 0%, 
    rgba(32, 39, 49, 0.95) 25%,
    rgba(45, 55, 72, 0.95) 50%,
    rgba(32, 39, 49, 0.95) 75%,
    rgba(22, 27, 34, 0.95) 100%
  );
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(48, 54, 61, 0.3);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Text Gradient for Brand */
.text-gradient {
  background: linear-gradient(135deg, theme('colors.neuron.glow'), theme('colors.neuron.glow-hover'));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Responsive container */
.navbar-container {
  max-width: 1440px;
}

/* Mobile menu animation */
.mobile-menu-enter-active,
.mobile-menu-leave-active {
  transition: all 0.3s ease;
}

.mobile-menu-enter-from,
.mobile-menu-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* Animation Performance */
@media (prefers-reduced-motion: reduce) {
  .animate-pulse,
  .animate-synapse-pulse {
    animation: none !important;
  }
}

/* Remove focus outlines from buttons */
button:focus {
  outline: none;
  box-shadow: none;
}
</style>
