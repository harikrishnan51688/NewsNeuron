<template>
  <div 
    class="neuron-logo relative inline-flex items-center justify-center"
    :class="sizeClasses"
  >
    <!-- Animated SVG Logo -->
    <svg 
      :width="svgSize" 
      :height="svgSize" 
      viewBox="0 0 100 100" 
      class="neuron-svg"
      :class="{ 'animate-synapse-pulse': animated }"
    >
      <!-- Background glow effect -->
      <defs>
        <filter id="neuron-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge> 
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
        
        <linearGradient id="neuron-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" :style="`stop-color:${primaryColor};stop-opacity:1`" />
          <stop offset="100%" :style="`stop-color:${secondaryColor};stop-opacity:1`" />
        </linearGradient>
      </defs>
      
      <!-- Main 'N' structure -->
      <g filter="url(#neuron-glow)">
        <!-- Left vertical line -->
        <line 
          x1="25" y1="20" 
          x2="25" y2="80" 
          stroke="url(#neuron-gradient)" 
          stroke-width="4" 
          stroke-linecap="round"
          class="neuron-line"
        />
        
        <!-- Right vertical line -->
        <line 
          x1="75" y1="20" 
          x2="75" y2="80" 
          stroke="url(#neuron-gradient)" 
          stroke-width="4" 
          stroke-linecap="round"
          class="neuron-line"
        />
        
        <!-- Diagonal synapse connection (enhanced) -->
        <line 
          x1="25" y1="30" 
          x2="75" y2="70" 
          stroke="url(#neuron-gradient)" 
          stroke-width="6" 
          stroke-linecap="round"
          class="synapse-connection"
        />
        
        <!-- Synapse nodes (circles at connection points) -->
        <circle 
          cx="25" cy="30" r="4" 
          fill="currentColor" 
          class="synapse-node animate-pulse"
        />
        <circle 
          cx="75" cy="70" r="4" 
          fill="currentColor" 
          class="synapse-node animate-pulse"
          style="animation-delay: 0.5s"
        />
        
        <!-- Additional connection nodes for neural network effect -->
        <circle 
          cx="50" cy="50" r="3" 
          fill="currentColor" 
          class="synapse-node animate-pulse"
          style="animation-delay: 0.25s"
        />
      </g>
      
      <!-- Animated particles flowing along the synapse -->
      <g v-if="showParticles">
        <circle r="2" fill="currentColor" class="flowing-particle">
          <animateMotion 
            dur="2s" 
            repeatCount="indefinite"
            path="M25,30 L50,50 L75,70"
          />
          <animate 
            attributeName="opacity" 
            values="0;1;1;0" 
            dur="2s" 
            repeatCount="indefinite"
          />
        </circle>
        
        <circle r="1.5" fill="currentColor" class="flowing-particle">
          <animateMotion 
            dur="2.5s" 
            repeatCount="indefinite"
            path="M25,30 L50,50 L75,70"
            begin="0.5s"
          />
          <animate 
            attributeName="opacity" 
            values="0;1;1;0" 
            dur="2.5s" 
            repeatCount="indefinite"
            begin="0.5s"
          />
        </circle>
      </g>
    </svg>
    
    <!-- Hover effect overlay -->
    <div 
      v-if="interactive"
      class="absolute inset-0 rounded-full bg-neuron-glow/10 opacity-0 hover:opacity-100 transition-opacity duration-300"
    ></div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

// Props
const props = defineProps({
  size: {
    type: String,
    default: 'md',
    validator: (value) => ['xs', 'sm', 'md', 'lg', 'xl'].includes(value)
  },
  animated: {
    type: Boolean,
    default: true
  },
  interactive: {
    type: Boolean,
    default: false
  },
  showParticles: {
    type: Boolean,
    default: false
  },
  primaryColor: {
    type: String,
    default: '#00A7E1' // neuron-glow
  },
  secondaryColor: {
    type: String,
    default: '#33BBF3' // neuron-glow-hover
  }
})

// Computed
const sizeClasses = computed(() => {
  switch (props.size) {
    case 'xs':
      return 'w-4 h-4'
    case 'sm':
      return 'w-6 h-6'
    case 'md':
      return 'w-8 h-8'
    case 'lg':
      return 'w-12 h-12'
    case 'xl':
      return 'w-16 h-16'
    default:
      return 'w-8 h-8'
  }
})

const svgSize = computed(() => {
  switch (props.size) {
    case 'xs':
      return 16
    case 'sm':
      return 24
    case 'md':
      return 32
    case 'lg':
      return 48
    case 'xl':
      return 64
    default:
      return 32
  }
})
</script>

<style scoped>
.neuron-logo {
  color: currentColor;
}

.neuron-svg {
  transition: all 0.3s ease;
}

.neuron-line {
  transition: all 0.3s ease;
}

.synapse-connection {
  filter: drop-shadow(0 0 6px currentColor);
}

.synapse-node {
  filter: drop-shadow(0 0 3px currentColor);
}

.flowing-particle {
  filter: drop-shadow(0 0 4px currentColor);
}

/* Hover effects */
.neuron-logo:hover .neuron-svg {
  transform: scale(1.05);
}

.neuron-logo:hover .synapse-connection {
  filter: drop-shadow(0 0 8px currentColor);
}

/* Enhanced pulse animation for synapse effect */
@keyframes synapse-pulse {
  0%, 100% { 
    transform: scale(1);
    filter: drop-shadow(0 0 6px currentColor);
  }
  50% { 
    transform: scale(1.05);
    filter: drop-shadow(0 0 12px currentColor);
  }
}

.animate-synapse-pulse {
  animation: synapse-pulse 2s ease-in-out infinite;
}

/* Custom pulse for nodes with staggered timing */
.synapse-node {
  animation: node-pulse 1.5s ease-in-out infinite;
}

@keyframes node-pulse {
  0%, 100% { 
    opacity: 0.8;
    transform: scale(1);
  }
  50% { 
    opacity: 1;
    transform: scale(1.2);
  }
}
</style>
