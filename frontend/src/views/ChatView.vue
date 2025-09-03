<template>
  <div class="chat-view min-h-screen bg-neuron-bg-primary">
    
    <!-- Subtle Background Particles -->
    <div class="fixed inset-0 overflow-hidden pointer-events-none">
      <div v-for="i in 8" :key="i" 
           :class="getParticleColor(i)"
           class="absolute w-0.5 h-0.5 rounded-full animate-pulse"
           :style="particleStyle(i)">
      </div>
    </div>

    <!-- Main Chat Interface - Simplified -->
    <div class="max-w-4xl mx-auto px-6 pb-6">
      
      <!-- Messages Container -->
      <div 
        ref="messagesContainer"
        class="min-h-[50vh] overflow-y-auto py-6 space-y-6 scroll-smooth"
      >
        
        <!-- Welcome Message (when no messages) -->
        <div v-if="messages.length === 0" 
             class="text-center py-4 space-y-3">
          <div class="w-16 h-16 bg-gradient-to-br from-accent-emerald to-accent-cyan rounded-full mx-auto flex items-center justify-center mb-3">
            <Brain class="w-8 h-8 text-white" />
          </div>
          <div>
            <h2 class="text-xl font-heading text-neuron-text-primary mb-2">How can I help you today?</h2>
            <p class="text-neuron-text-secondary">Ask me anything about the news or explore recent events</p>
          </div>
          
          <!-- Quick Suggestions - Smaller and closer -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-xl mx-auto mt-2">
            <button
              v-for="suggestion in quickSuggestions"
              :key="suggestion"
              @click="sendMessage(suggestion)"
              class="p-3 text-center bg-neuron-bg-content/50 border border-neuron-border/50 rounded-lg hover:border-accent-emerald/50 hover:bg-neuron-bg-content transition-all group text-sm"
            >
              <span class="text-neuron-text-primary group-hover:text-accent-emerald transition-colors">{{ suggestion }}</span>
            </button>
          </div>
        </div>

        <!-- Chat Messages -->
        <div v-for="(message, index) in messages" :key="index" class="message-container">
          
          <!-- User Message -->
          <div v-if="message.role === 'user'" class="flex justify-end mb-4">
            <div class="max-w-xs sm:max-w-md lg:max-w-lg">
              <div class="bg-gradient-to-br from-accent-violet to-accent-rose text-white p-4 rounded-2xl rounded-br-sm shadow-sm">
                <p class="text-sm leading-relaxed">{{ message.content }}</p>
              </div>
            </div>
          </div>

          <!-- AI Message -->
          <div v-else class="flex items-start space-x-3 mb-4">
            <div class="w-8 h-8 bg-gradient-to-br from-accent-emerald to-accent-cyan rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <Brain class="w-4 h-4 text-white" />
            </div>
            <div class="max-w-xs sm:max-w-md lg:max-w-lg">
              <div class="bg-neuron-bg-content border border-neuron-border p-4 rounded-2xl rounded-tl-sm shadow-sm">
                <p class="text-sm leading-relaxed text-neuron-text-primary whitespace-pre-wrap">{{ message.content }}</p>
                
                <!-- Processing time (subtle) -->
                <div v-if="message.processingTime" class="text-xs text-neuron-text-secondary/50 mt-1">
                  {{ message.processingTime }}
                </div>
                
                <!-- Copy Button -->
                <button
                  @click="copyMessage(message.content)"
                  class="mt-2 p-1 hover:bg-neuron-bg-primary rounded transition-colors"
                  title="Copy message"
                >
                  <Copy class="w-3 h-3 text-neuron-text-secondary" />
                </button>
              </div>
              
              <!-- Suggested Follow-ups (for latest AI message) -->
              <div v-if="message.suggestions && index === messages.length - 1" class="mt-3 space-y-1">
                <button
                  v-for="suggestion in message.suggestions"
                  :key="suggestion"
                  @click="sendMessage(suggestion)"
                  class="block w-full text-left px-3 py-2 bg-neuron-bg-primary/30 border border-neuron-border/30 rounded-lg text-xs text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/30 transition-all"
                >
                  {{ suggestion }}
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Typing Indicator -->
        <div v-if="isTyping" class="flex items-start space-x-3 mb-4">
          <div class="w-8 h-8 bg-gradient-to-br from-accent-emerald to-accent-cyan rounded-full flex items-center justify-center flex-shrink-0">
            <Brain class="w-4 h-4 text-white" />
          </div>
          <div class="bg-neuron-bg-content border border-neuron-border p-4 rounded-2xl rounded-tl-sm">
            <TypingIndicator />
          </div>
        </div>
      </div>

      <!-- Input Area - Streamlined -->
      <div class="bg-neuron-bg-primary/80 backdrop-blur-sm py-2">
        <div class="bg-neuron-bg-content border border-neuron-border rounded-2xl shadow-sm">
          <form @submit.prevent="sendUserMessage" class="flex items-end p-2">
            <textarea
              v-model="userInput"
              ref="messageInput"
              @keydown="handleKeyDown"
              placeholder="Ask me anything..."
              class="flex-1 bg-transparent border-0 p-3 text-neuron-text-primary placeholder-neuron-text-secondary resize-none focus:outline-none min-h-[2.5rem] max-h-32"
              rows="1"
            ></textarea>
            
            <button
              type="submit"
              :disabled="!canSend"
              class="ml-2 p-3 bg-gradient-to-r from-accent-emerald to-accent-cyan text-white rounded-xl hover:shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send v-if="!isLoading" class="w-4 h-4" />
              <Loader2 v-else class="w-4 h-4 animate-spin" />
            </button>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, nextTick, onMounted } from 'vue'
import { Brain, Send, Loader2, Copy } from 'lucide-vue-next'
import TypingIndicator from '@/components/ui/TypingIndicator.vue'

export default {
  name: 'ChatView',
  components: {
    Brain,
    Send,
    Loader2,
    Copy,
    TypingIndicator
  },
  setup() {
    const messages = ref([])
    const userInput = ref('')
    const isLoading = ref(false)
    const isTyping = ref(false)
    const messagesContainer = ref(null)
    const messageInput = ref(null)

    const quickSuggestions = [
      'Latest tech news today',
      'Current market trends', 
      'Breaking political updates',
      'Recent AI developments'
    ]

    const canSend = computed(() => {
      return userInput.value.trim().length > 0 && !isLoading.value
    })

    const sendMessage = async (content) => {
      if (!content || content.trim().length === 0) return
      
      // Add user message
      messages.value.push({
        role: 'user',
        content: content.trim(),
        timestamp: new Date()
      })

      // Clear input if it was from the input field
      if (content === userInput.value.trim()) {
        userInput.value = ''
      }

      isLoading.value = true
      isTyping.value = true

      try {
        // Simulate API call with realistic timing
        const startTime = Date.now()
        await new Promise(resolve => setTimeout(resolve, 1500))
        const processingTime = ((Date.now() - startTime) / 1000).toFixed(1)
        
        // Add AI response with smart suggestions
        const aiResponse = {
          role: 'assistant',
          content: 'I understand your question about "' + content + '". Let me help you with that. This is a sample response to demonstrate the new minimalistic chat interface with backend-inspired features.',
          timestamp: new Date(),
          suggestions: generateSmartSuggestions(content),
          processingTime: processingTime + 's'
        }
        
        messages.value.push(aiResponse)
      } catch (error) {
        console.error('Error sending message:', error)
        messages.value.push({
          role: 'assistant',
          content: 'I apologize, but I encountered an error processing your request. Please try again.',
          timestamp: new Date()
        })
      } finally {
        isLoading.value = false
        isTyping.value = false
        await scrollToBottom()
      }
    }

    const sendUserMessage = () => {
      if (canSend.value) {
        sendMessage(userInput.value)
      }
    }

    const copyMessage = async (content) => {
      try {
        await navigator.clipboard.writeText(content)
      } catch (error) {
        console.error('Failed to copy message:', error)
      }
    }

    const scrollToBottom = async () => {
      await nextTick()
      if (messagesContainer.value) {
        messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
      }
    }

    const handleKeyDown = (event) => {
      if (event.key === 'Enter') {
        if (event.shiftKey) {
          // Allow new line with Shift+Enter
          return
        } else {
          // Send message with Enter
          event.preventDefault()
          sendUserMessage()
        }
      }
    }

    // Smart suggestion generation based on user input
    const generateSmartSuggestions = (userQuery) => {
      const suggestions = {
        'tech': ['What companies are leading this?', 'Recent developments?', 'Market impact?'],
        'political': ['Key stakeholders involved?', 'Public reaction?', 'Policy implications?'],
        'market': ['Stock performance?', 'Expert predictions?', 'Global impact?'],
        'ai': ['Technical details?', 'Ethical considerations?', 'Future prospects?'],
        'science': ['Research findings?', 'Practical applications?', 'Next steps?']
      }
      
      const query = userQuery.toLowerCase()
      
      if (query.includes('tech') || query.includes('technology')) return suggestions.tech
      if (query.includes('politic') || query.includes('government')) return suggestions.political
      if (query.includes('market') || query.includes('economy')) return suggestions.market
      if (query.includes('ai') || query.includes('artificial')) return suggestions.ai
      if (query.includes('science') || query.includes('research')) return suggestions.science
      
      // Default suggestions
      return ['Tell me more', 'Any sources?', 'Related topics?']
    }

    // Particle animation helpers
    const getParticleColor = (index) => {
      const colors = [
        'bg-accent-emerald/20',
        'bg-accent-cyan/20',
        'bg-accent-violet/20',
        'bg-accent-rose/20',
        'bg-accent-amber/20',
        'bg-accent-orange/20'
      ]
      return colors[index % colors.length]
    }

    const particleStyle = (index) => {
      return {
        left: `${(index * 13 + 10) % 100}%`,
        top: `${(index * 17 + 20) % 100}%`,
        animationDelay: `${index * 0.5}s`,
        animationDuration: `${3 + (index % 3)}s`
      }
    }

    onMounted(() => {
      messageInput.value?.focus()
    })

    return {
      messages,
      userInput,
      isLoading,
      isTyping,
      messagesContainer,
      messageInput,
      quickSuggestions,
      canSend,
      sendMessage,
      sendUserMessage,
      copyMessage,
      handleKeyDown,
      generateSmartSuggestions,
      getParticleColor,
      particleStyle
    }
  }
}
</script>

<style scoped>
.chat-view {
  min-height: 100vh;
}

/* Custom scrollbar */
.overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}

.overflow-y-auto::-webkit-scrollbar-track {
  background: transparent;
}

.overflow-y-auto::-webkit-scrollbar-thumb {
  background: rgba(156, 163, 175, 0.3);
  border-radius: 3px;
}

.overflow-y-auto::-webkit-scrollbar-thumb:hover {
  background: rgba(156, 163, 175, 0.5);
}

/* Auto-resize textarea */
textarea {
  field-sizing: content;
}

/* Smooth animations */
.message-container {
  animation: fadeInUp 0.3s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Button styles */
.btn-icon {
  padding: 0.5rem;
  border-radius: 0.5rem;
  color: rgb(156 163 175);
  transition: all 0.15s ease-in-out;
}

.btn-icon:hover {
  background-color: rgba(255, 255, 255, 0.05);
}
</style>
