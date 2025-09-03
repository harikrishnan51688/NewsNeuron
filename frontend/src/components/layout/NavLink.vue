<template>
  <router-link
    :to="to"
    class="nav-link group flex items-center transition-all duration-200 rounded-lg"
    :class="linkClasses"
    active-class="active"
  >
    
    <!-- Icon -->
    <component 
      :is="icon" 
      class="flex-shrink-0 transition-all duration-200"
      :class="iconClasses"
    />
    
    <!-- Label -->
    <span 
      v-if="showLabel" 
      class="font-heading font-medium transition-all duration-200"
      :class="labelClasses"
    >
      <slot />
    </span>
    
    <!-- Badge -->
    <div 
      v-if="badge && showBadge" 
      class="flex-shrink-0"
      :class="{ 'ml-auto': !horizontal && !mobile, 'ml-2': horizontal || mobile }"
    >
      <span 
        class="inline-flex items-center justify-center px-2 py-1 text-xs font-medium rounded-full transition-all duration-200"
        :class="badgeClasses"
      >
        {{ badge }}
      </span>
    </div>
  </router-link>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

// Props
const props = defineProps({
  to: {
    type: String,
    required: true
  },
  icon: {
    type: Object,
    required: true
  },
  badge: {
    type: String,
    default: null
  },
  collapsed: {
    type: Boolean,
    default: false
  },
  horizontal: {
    type: Boolean,
    default: false
  },
  mobile: {
    type: Boolean,
    default: false
  }
})

// Router
const route = useRoute()

// Computed
const isActive = computed(() => {
  if (props.to === '/') {
    return route.path === '/'
  }
  return route.path.startsWith(props.to)
})

const showLabel = computed(() => {
  return !props.collapsed || props.horizontal || props.mobile
})

const showBadge = computed(() => {
  return props.badge && (!props.collapsed || props.horizontal || props.mobile)
})

const linkClasses = computed(() => {
  const base = 'hover:bg-neuron-bg-primary hover:text-neuron-text-primary'
  
  if (props.mobile) {
    return [
      base,
      'flex-col space-y-1 p-3 text-center',
      isActive.value ? 'bg-neuron-glow/10 border border-neuron-glow/20' : ''
    ]
  }
  
  if (props.horizontal) {
    return [
      base,
      'px-4 py-2.5',
      isActive.value ? 'bg-neuron-glow/10 border border-neuron-glow/20' : ''
    ]
  }
  
  if (props.collapsed) {
    return [
      base,
      'px-3 py-3 justify-center relative',
      isActive.value ? 'bg-neuron-glow/10 border border-neuron-glow/20' : ''
    ]
  }
  
  return [
    base,
    'px-4 py-3',
    isActive.value ? 'bg-neuron-glow/10 border border-neuron-glow/20' : ''
  ]
})

const iconClasses = computed(() => {
  const base = 'transition-all duration-200'
  const color = isActive.value ? 'text-neuron-glow' : 'text-neuron-text-secondary group-hover:text-neuron-text-primary'
  
  if (props.mobile) {
    return [base, 'w-5 h-5', color]
  }
  
  if (props.horizontal) {
    return [base, 'w-5 h-5 mr-2', color]
  }
  
  if (props.collapsed) {
    return [base, 'w-5 h-5', color]
  }
  
  return [base, 'w-5 h-5 mr-3', color]
})

const labelClasses = computed(() => {
  const color = isActive.value ? 'text-neuron-glow' : 'text-neuron-text-secondary group-hover:text-neuron-text-primary'
  
  if (props.mobile) {
    return ['text-xs', color]
  }
  
  if (props.horizontal) {
    return ['text-base', color]
  }
  
  return ['text-sm flex-1', color]
})

const badgeClasses = computed(() => {
  return isActive.value 
    ? 'bg-neuron-glow text-white'
    : 'bg-neuron-text-secondary/20 text-neuron-text-secondary group-hover:bg-neuron-glow/20 group-hover:text-neuron-glow'
})
</script>

<style scoped>
.nav-link {
  position: relative;
}

.nav-link.active {
  background: rgba(0, 167, 225, 0.1);
  border: 1px solid rgba(0, 167, 225, 0.2);
}

.nav-link.active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 20px;
  background: theme('colors.neuron.glow');
  border-radius: 0 2px 2px 0;
}

/* Horizontal navigation doesn't need the left indicator */
.nav-link.active.horizontal::before {
  display: none;
}

/* Mobile navigation gets a bottom indicator */
.nav-link.active.mobile::before {
  top: auto;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 20px;
  height: 3px;
  border-radius: 2px 2px 0 0;
}

/* Hover effects */
.nav-link:hover {
  transform: translateY(-1px);
}

.nav-link.mobile:hover {
  transform: scale(1.05);
}

/* Animation performance */
@media (prefers-reduced-motion: reduce) {
  .nav-link {
    transform: none !important;
    transition: none !important;
  }
}
</style>
