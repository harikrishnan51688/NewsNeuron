<template>
  <Teleport to="body">
    <div
      v-if="isOpen"
      class="fixed inset-0 z-40"
      @click="$emit('close')"
    >
      <div
        class="absolute right-4 top-16 w-96 bg-dark-card border border-dark-border rounded-2xl shadow-2xl overflow-hidden"
        @click.stop
      >
        <!-- Header -->
        <div class="px-6 py-4 border-b border-dark-border">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold text-text-primary">Notifications</h3>
            <div class="flex items-center space-x-3">
              <button
                v-if="unreadCount > 0"
                @click="markAllRead"
                class="text-sm text-neuron-glow hover:text-neuron-glow/80 transition-colors"
              >
                Mark all read
              </button>
              <button
                class="p-1 rounded-lg hover:bg-dark-background transition-colors"
                @click="$emit('close')"
              >
                <XIcon class="w-4 h-4 text-text-secondary" />
              </button>
            </div>
          </div>

          <!-- Summary Stats -->
          <div class="flex items-center mt-3 space-x-4 text-sm text-text-secondary">
            <span class="flex items-center">
              <div class="w-2 h-2 bg-neuron-glow rounded-full mr-2"></div>
              {{ unreadCount }} unread
            </span>
            <span>{{ totalCount }} total</span>
          </div>
        </div>

        <!-- Filters -->
        <div class="px-6 py-3 border-b border-dark-border">
          <div class="flex space-x-2">
            <button
              v-for="filter in filters"
              :key="filter.id"
              :class="[
                'px-3 py-1 rounded-lg text-sm font-medium transition-all',
                activeFilter === filter.id
                  ? 'bg-neuron-glow/20 text-neuron-glow border border-neuron-glow/30'
                  : 'bg-dark-background border border-dark-border text-text-secondary hover:text-text-primary hover:border-neuron-glow/50'
              ]"
              @click="activeFilter = filter.id"
            >
              <component :is="filter.icon" class="w-3 h-3 mr-1 inline" />
              {{ filter.label }}
            </button>
          </div>
        </div>

        <!-- Notifications List -->
        <div class="max-h-96 overflow-y-auto">
          <div
            v-if="filteredNotifications.length === 0"
            class="p-8 text-center"
          >
            <BellIcon class="w-12 h-12 mx-auto mb-4 text-text-muted" />
            <h4 class="text-sm font-medium text-text-secondary mb-2">No notifications</h4>
            <p class="text-xs text-text-muted">You're all caught up!</p>
          </div>

          <div v-else class="py-2">
            <div
              v-for="notification in filteredNotifications"
              :key="notification.id"
              :class="[
                'px-6 py-4 border-l-2 transition-all cursor-pointer',
                notification.read
                  ? 'border-transparent hover:border-dark-border hover:bg-dark-background/30'
                  : 'border-neuron-glow/50 bg-neuron-glow/5 hover:bg-neuron-glow/10'
              ]"
              @click="handleNotificationClick(notification)"
            >
              <div class="flex items-start space-x-3">
                <!-- Icon -->
                <div class="flex-shrink-0">
                  <div
                    :class="[
                      'w-10 h-10 rounded-lg flex items-center justify-center',
                      getNotificationStyle(notification.type)
                    ]"
                  >
                    <component
                      :is="getNotificationIcon(notification.type)"
                      class="w-5 h-5"
                    />
                  </div>
                </div>

                <!-- Content -->
                <div class="flex-1 min-w-0">
                  <div class="flex items-start justify-between">
                    <h4 class="text-sm font-medium text-text-primary truncate pr-2">
                      {{ notification.title }}
                    </h4>
                    <div class="flex items-center space-x-2">
                      <span class="text-xs text-text-muted">{{ formatTime(notification.timestamp) }}</span>
                      <div
                        v-if="!notification.read"
                        class="w-2 h-2 bg-neuron-glow rounded-full"
                      ></div>
                    </div>
                  </div>
                  
                  <p class="text-sm text-text-secondary mt-1 line-clamp-2">
                    {{ notification.message }}
                  </p>

                  <!-- Tags -->
                  <div v-if="notification.tags" class="flex items-center mt-2 space-x-2">
                    <span
                      v-for="tag in notification.tags"
                      :key="tag"
                      class="px-2 py-1 bg-dark-background border border-dark-border rounded text-xs text-text-muted"
                    >
                      {{ tag }}
                    </span>
                  </div>

                  <!-- Actions -->
                  <div v-if="notification.actions" class="flex items-center mt-3 space-x-2">
                    <button
                      v-for="action in notification.actions"
                      :key="action.id"
                      :class="[
                        'px-3 py-1 rounded-lg text-xs font-medium transition-all',
                        action.primary
                          ? 'bg-neuron-glow text-white hover:bg-neuron-glow/80'
                          : 'bg-dark-background border border-dark-border text-text-secondary hover:text-text-primary hover:border-neuron-glow/50'
                      ]"
                      @click.stop="handleAction(action, notification)"
                    >
                      {{ action.label }}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="px-6 py-3 border-t border-dark-border bg-dark-background/50">
          <button
            class="w-full text-center text-sm text-neuron-glow hover:text-neuron-glow/80 transition-colors"
            @click="$emit('view-all')"
          >
            View all notifications
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, computed } from 'vue'
import { 
  BellIcon, 
  XMarkIcon as XIcon, 
  InboxIcon, 
  SparklesIcon, 
  ArrowTrendingUpIcon as TrendingUpIcon, 
  NewspaperIcon,
  UserIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon as InfoIcon
} from '@heroicons/vue/24/outline'

// Props & Emits
const props = defineProps({
  isOpen: {
    type: Boolean,
    default: false
  },
  notifications: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['close', 'mark-read', 'mark-all-read', 'view-all', 'action'])

// Data
const activeFilter = ref('all')

const filters = [
  { id: 'all', label: 'All', icon: InboxIcon },
  { id: 'ai', label: 'AI', icon: SparklesIcon },
  { id: 'news', label: 'News', icon: NewspaperIcon },
  { id: 'trends', label: 'Trends', icon: TrendingUpIcon }
]

// Mock notifications for demo
const defaultNotifications = [
  {
    id: 1,
    type: 'ai',
    title: 'New AI Analysis Ready',
    message: 'Your requested analysis on "Climate Change Policy" has been completed with 15 new insights.',
    timestamp: new Date(Date.now() - 5 * 60 * 1000),
    read: false,
    tags: ['Analysis', 'Climate'],
    actions: [
      { id: 'view', label: 'View Report', primary: true },
      { id: 'dismiss', label: 'Dismiss', primary: false }
    ]
  },
  {
    id: 2,
    type: 'news',
    title: 'Breaking News Alert',
    message: 'Major development in renewable energy sector affects 3 entities in your watchlist.',
    timestamp: new Date(Date.now() - 15 * 60 * 1000),
    read: false,
    tags: ['Breaking', 'Energy']
  },
  {
    id: 3,
    type: 'trend',
    title: 'Trending Topic Detected',
    message: 'Artificial Intelligence is trending with 340% increase in mentions.',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    read: true,
    tags: ['AI', 'Trending']
  },
  {
    id: 4,
    type: 'system',
    title: 'System Update Complete',
    message: 'NewsNeuron has been updated with improved entity recognition and faster search.',
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000),
    read: true,
    tags: ['Update']
  }
]

// Computed
const allNotifications = computed(() => props.notifications.length ? props.notifications : defaultNotifications)

const filteredNotifications = computed(() => {
  if (activeFilter.value === 'all') {
    return allNotifications.value
  }
  return allNotifications.value.filter(n => n.type === activeFilter.value)
})

const unreadCount = computed(() => 
  allNotifications.value.filter(n => !n.read).length
)

const totalCount = computed(() => allNotifications.value.length)

// Methods
const markAllRead = () => {
  emit('mark-all-read')
}

const handleNotificationClick = (notification) => {
  if (!notification.read) {
    emit('mark-read', notification.id)
  }
}

const handleAction = (action, notification) => {
  emit('action', { action, notification })
}

const getNotificationIcon = (type) => {
  const icons = {
    ai: SparklesIcon,
    news: NewspaperIcon,
    trend: TrendingUpIcon,
    user: UserIcon,
    warning: ExclamationTriangleIcon,
    success: CheckCircleIcon,
    info: InfoIcon,
    system: InfoIcon
  }
  return icons[type] || BellIcon
}

const getNotificationStyle = (type) => {
  const styles = {
    ai: 'bg-purple-500/20 text-purple-400',
    news: 'bg-blue-500/20 text-blue-400',
    trend: 'bg-green-500/20 text-green-400',
    user: 'bg-neuron-glow/20 text-neuron-glow',
    warning: 'bg-yellow-500/20 text-yellow-400',
    success: 'bg-green-500/20 text-green-400',
    info: 'bg-blue-500/20 text-blue-400',
    system: 'bg-gray-500/20 text-gray-400'
  }
  return styles[type] || 'bg-neuron-glow/20 text-neuron-glow'
}

const formatTime = (timestamp) => {
  const now = new Date()
  const diff = now - timestamp
  const minutes = Math.floor(diff / (1000 * 60))
  const hours = Math.floor(diff / (1000 * 60 * 60))
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))

  if (minutes < 1) return 'Just now'
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days < 7) return `${days}d ago`
  
  return timestamp.toLocaleDateString()
}
</script>

