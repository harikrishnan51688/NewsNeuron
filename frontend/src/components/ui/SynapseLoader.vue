<template>
  <div 
    class="synapse-loader relative inline-flex items-center justify-center"
    :class="sizeClasses"
    role="status"
    :aria-label="ariaLabel"
  >
    
    <!-- Main synapse structure -->
    <svg 
      :width="svgSize" 
      :height="svgSize" 
      viewBox="0 0 100 100" 
      class="synapse-svg"
    >
      <defs>
        <!-- Glowing gradient for the synapse -->
        <linearGradient id="synapse-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#00A7E1" stop-opacity="1" />
          <stop offset="50%" stop-color="#33BBF3" stop-opacity="0.8" />
          <stop offset="100%" stop-color="#00A7E1" stop-opacity="1" />
        </linearGradient>
        
        <!-- Glow filter -->
        <filter id="synapse-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
          <feMerge> 
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      
      <!-- Background circle -->
      <circle 
        cx="50" cy="50" r="45" 
        fill="none" 
        stroke="rgba(48, 54, 61, 0.3)" 
        stroke-width="2"
        class="background-circle"
      />
      
      <!-- Animated synapse connections -->
      <g filter="url(#synapse-glow)">
        <!-- Primary connection -->
        <line 
          x1="20" y1="35" 
          x2="80" y2="65" 
          stroke="url(#synapse-gradient)" 
          stroke-width="3" 
          stroke-linecap="round"
          class="synapse-line primary-line"
        />
        
        <!-- Secondary connections -->
        <line 
          x1="30" y1="20" 
          x2="70" y2="80" 
          stroke="url(#synapse-gradient)" 
          stroke-width="2" 
          stroke-linecap="round"
          class="synapse-line secondary-line"
          stroke-opacity="0.7"
        />
        
        <line 
          x1="15" y1="60" 
          x2="85" y2="40" 
          stroke="url(#synapse-gradient)" 
          stroke-width="2" 
          stroke-linecap="round"
          class="synapse-line secondary-line"
          stroke-opacity="0.7"
        />
      </g>
      
      <!-- Animated nodes -->
      <g class="synapse-nodes">
        <!-- Center node -->
        <circle 
          cx="50" cy="50" r="6" 
          fill="url(#synapse-gradient)" 
          class="center-node"
        />
        
        <!-- Connection nodes -->
        <circle 
          cx="20" cy="35" r="4" 
          fill="#00A7E1" 
          class="connection-node node-1"
        />
        <circle 
          cx="80" cy="65" r="4" 
          fill="#00A7E1" 
          class="connection-node node-2"
        />
        <circle 
          cx="30" cy="20" r="3" 
          fill="#33BBF3" 
          class="connection-node node-3"
        />
        <circle 
          cx="70" cy="80" r="3" 
          fill="#33BBF3" 
          class="connection-node node-4"
        />
        <circle 
          cx="15" cy="60" r="3" 
          fill="#33BBF3" 
          class="connection-node node-5"
        />
        <circle 
          cx="85" cy="40" r="3" 
          fill="#33BBF3" 
          class="connection-node node-6"
        />
      </g>
      
      <!-- Flowing particles -->
      <g class="flowing-particles">
        <!-- Particle 1 -->
        <circle r="2" fill="#00A7E1" class="particle">
          <animateMotion 
            dur="3s" 
            repeatCount="indefinite"
            path="M20,35 Q35,25 50,50 Q65,75 80,65"
          />
          <animate 
            attributeName="opacity" 
            values="0;1;1;0" 
            dur="3s" 
            repeatCount="indefinite"
          />
        </circle>
        
        <!-- Particle 2 -->
        <circle r="1.5" fill="#33BBF3" class="particle">
          <animateMotion 
            dur="2.5s" 
            repeatCount="indefinite"
            path="M30,20 Q40,35 50,50 Q60,65 70,80"
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
        
        <!-- Particle 3 -->
        <circle r="1.5" fill="#33BBF3" class="particle">
          <animateMotion 
            dur="3.5s" 
            repeatCount="indefinite"
            path="M15,60 Q30,55 50,50 Q70,45 85,40"
            begin="1s"
          />
          <animate 
            attributeName="opacity" 
            values="0;1;1;0" 
            dur="3.5s" 
            repeatCount="indefinite"
            begin="1s"
          />
        </circle>
      </g>
    </svg>
    
    <!-- Loading text (optional) -->
    <div 
      v-if="showText" 
      class="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-neuron-text-secondary text-sm font-heading"
    >
      {{ loadingText }}
    </div>
    
  </div>
</template>

<script setup>
import { computed } from 'vue'

// Props
const props = defineProps({
  size: {
    type: String,
    default: 'md',
    validator: (value) => ['sm', 'md', 'lg', 'xl'].includes(value)
  },
  showText: {
    type: Boolean,
    default: false
  },
  loadingText: {
    type: String,
    default: 'Processing...'
  },
  ariaLabel: {
    type: String,
    default: 'Loading content'
  }
})

// Computed
const sizeClasses = computed(() => {
  switch (props.size) {
    case 'sm':
      return 'w-8 h-8'
    case 'md':
      return 'w-12 h-12'
    case 'lg':
      return 'w-16 h-16'
    case 'xl':
      return 'w-24 h-24'
    default:
      return 'w-12 h-12'
  }
})

const svgSize = computed(() => {
  switch (props.size) {
    case 'sm':
      return 32
    case 'md':
      return 48
    case 'lg':
      return 64
    case 'xl':
      return 96
    default:
      return 48
  }
})
</script>

<style scoped>
.synapse-loader {
  color: #00A7E1;
}

.synapse-svg {
  animation: rotate 4s linear infinite;
}

@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Synapse line animations */
.primary-line {
  animation: primary-pulse 2s ease-in-out infinite;
}

.secondary-line {
  animation: secondary-pulse 2.5s ease-in-out infinite;
}

@keyframes primary-pulse {
  0%, 100% { 
    stroke-width: 3;
    stroke-opacity: 1;
  }
  50% { 
    stroke-width: 4;
    stroke-opacity: 0.8;
  }
}

@keyframes secondary-pulse {
  0%, 100% { 
    stroke-width: 2;
    stroke-opacity: 0.7;
  }
  50% { 
    stroke-width: 3;
    stroke-opacity: 0.9;
  }
}

/* Node animations */
.center-node {
  animation: center-pulse 1.5s ease-in-out infinite;
}

.connection-node {
  animation: node-pulse 2s ease-in-out infinite;
}

.node-1 { animation-delay: 0s; }
.node-2 { animation-delay: 0.3s; }
.node-3 { animation-delay: 0.6s; }
.node-4 { animation-delay: 0.9s; }
.node-5 { animation-delay: 1.2s; }
.node-6 { animation-delay: 1.5s; }

@keyframes center-pulse {
  0%, 100% { 
    r: 6;
    opacity: 1;
  }
  50% { 
    r: 8;
    opacity: 0.8;
  }
}

@keyframes node-pulse {
  0%, 100% { 
    opacity: 0.6;
    transform: scale(1);
  }
  50% { 
    opacity: 1;
    transform: scale(1.3);
  }
}

/* Particle effects */
.particle {
  filter: drop-shadow(0 0 4px currentColor);
}

/* Background circle subtle animation */
.background-circle {
  animation: background-fade 3s ease-in-out infinite;
}

@keyframes background-fade {
  0%, 100% { stroke-opacity: 0.3; }
  50% { stroke-opacity: 0.5; }
}

/* Accessibility: Respect reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  .synapse-svg {
    animation: none;
  }
  
  .primary-line,
  .secondary-line,
  .center-node,
  .connection-node,
  .background-circle {
    animation: none;
  }
  
  .flowing-particles {
    display: none;
  }
}
</style>
