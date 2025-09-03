<template>
  <div class="chat-view min-h-screen bg-neuron-bg-primary">

    <!-- Subtle Background Particles -->
    <div class="fixed inset-0 overflow-hidden pointer-events-none">
      <div v-for="i in 8" :key="i" :class="getParticleColor(i)" class="absolute w-0.5 h-0.5 rounded-full animate-pulse"
        :style="particleStyle(i)">
      </div>
    </div>

    <!-- Main Chat Interface -->
    <div class="max-w-4xl mx-auto px-6 pb-6">

      <!-- Messages Container -->
      <div ref="messagesContainer" class="min-h-[50vh] overflow-y-auto py-6 space-y-6 scroll-smooth">

        <!-- Welcome Message (when no messages) -->
        <div v-if="messages.length === 0" class="text-center py-4 space-y-3">
          <div
            class="w-16 h-16 bg-gradient-to-br from-accent-emerald to-accent-cyan rounded-full mx-auto flex items-center justify-center mb-3">
            <Brain class="w-8 h-8 text-white" />
          </div>
          <div>
            <h2 class="text-xl font-heading text-neuron-text-primary mb-2">How can I help you today?</h2>
            <p class="text-neuron-text-secondary">Ask me anything about the news or explore recent events</p>
          </div>

          <!-- Quick Suggestions -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-xl mx-auto mt-2">
            <button v-for="suggestion in quickSuggestions" :key="suggestion" @click="sendMessage(suggestion)"
              class="p-3 text-center bg-neuron-bg-content/50 border border-neuron-border/50 rounded-lg hover:border-accent-emerald/50 hover:bg-neuron-bg-content transition-all group text-sm">
              <span class="text-neuron-text-primary group-hover:text-accent-emerald transition-colors">{{ suggestion
              }}</span>
            </button>
          </div>
        </div>

        <!-- Chat Messages -->
        <div v-for="(message, index) in messages" :key="index" class="message-container">

          <!-- User Message -->
          <div v-if="message.role === 'user'" class="flex justify-end mb-4">
            <div class="max-w-xs sm:max-w-md lg:max-w-2xl">
              <div
                class="bg-gradient-to-br from-accent-violet to-accent-rose text-white p-4 rounded-2xl rounded-br-sm shadow-sm">
                <p class="text-sm leading-relaxed">{{ message.content }}</p>
              </div>
            </div>
          </div>

          <!-- AI Message -->
          <div v-else class="flex items-start space-x-3 mb-4">
            <div
              class="w-8 h-8 bg-gradient-to-br from-accent-emerald to-accent-cyan rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <Brain class="w-4 h-4 text-white" />
            </div>
            <div class="max-w-xs sm:max-w-md lg:max-w-3xl">
              <div class="bg-neuron-bg-content border border-neuron-border p-5 rounded-2xl rounded-tl-sm shadow-sm">
                <!-- Formatted Response Content -->
                <div class="prose prose-sm max-w-none text-neuron-text-primary">
                  <div v-html="formatResponse(message.content)" class="formatted-response"></div>
                  <span v-if="message.isStreaming" class="inline-block w-2 h-4 bg-accent-emerald animate-pulse ml-1"></span>
                </div>

                <!-- Processing time and actions -->
                <div class="flex items-center justify-between mt-3 pt-2 border-t border-neuron-border/30">
                  <div v-if="message.processingTime" class="text-xs text-neuron-text-secondary/70">
                    {{ message.processingTime }}
                  </div>
                  <div class="flex items-center space-x-2">
                    <button @click="copyMessage(message.content)"
                      class="p-1 hover:bg-neuron-bg-primary rounded transition-colors" title="Copy message">
                      <Copy class="w-3 h-3 text-neuron-text-secondary hover:text-accent-emerald" />
                    </button>
                    <button @click="regenerateResponse(index)"
                      class="p-1 hover:bg-neuron-bg-primary rounded transition-colors" title="Regenerate response">
                      <RotateCcw class="w-3 h-3 text-neuron-text-secondary hover:text-accent-emerald" />
                    </button>
                  </div>
                </div>
              </div>

              <!-- Suggested Follow-ups -->
              <div v-if="message.suggestions && index === messages.length - 1 && !message.isStreaming" class="mt-3 space-y-1">
                <button v-for="suggestion in message.suggestions" :key="suggestion" @click="sendMessage(suggestion)"
                  class="block w-full text-left px-3 py-2 bg-neuron-bg-primary/30 border border-neuron-border/30 rounded-lg text-xs text-neuron-text-secondary hover:text-accent-emerald hover:border-accent-emerald/30 transition-all">
                  {{ suggestion }}
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Typing Indicator -->
        <div v-if="isTyping" class="flex items-start space-x-3 mb-4">
          <div
            class="w-8 h-8 bg-gradient-to-br from-accent-emerald to-accent-cyan rounded-full flex items-center justify-center flex-shrink-0">
            <Brain class="w-4 h-4 text-white" />
          </div>
          <div class="bg-neuron-bg-content border border-neuron-border p-4 rounded-2xl rounded-tl-sm">
            <TypingIndicator />
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="bg-neuron-bg-primary/80 backdrop-blur-sm py-2">
        <div class="bg-neuron-bg-content border border-neuron-border rounded-2xl shadow-sm">
          <form @submit.prevent="sendUserMessage" class="flex items-end p-2">
            <textarea v-model="userInput" ref="messageInput" @keydown="handleKeyDown" placeholder="Ask me anything..."
              class="flex-1 bg-transparent border-0 p-3 text-neuron-text-primary placeholder-neuron-text-secondary resize-none focus:outline-none min-h-[2.5rem] max-h-32"
              rows="1"></textarea>

            <button type="submit" :disabled="!canSend"
              class="ml-2 p-3 bg-gradient-to-r from-accent-emerald to-accent-cyan text-white rounded-xl hover:shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed">
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
import TypingIndicator from '@/components/ui/TypingIndicator.vue'
import { Brain, Copy, Loader2, RotateCcw, Send } from 'lucide-vue-next'
import { computed, nextTick, onMounted, ref } from 'vue'

export default {
  name: 'ChatView',
  components: {
    Brain,
    Send,
    Loader2,
    Copy,
    RotateCcw,
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

    // Enhanced response formatting function
    const formatResponse = (text) => {
      if (!text) return ''
      
      let formatted = text
      
      // Handle code blocks with syntax highlighting
      formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)\n```/g, (match, lang, code) => {
        return `<div class="code-block bg-gray-900 rounded-lg p-4 my-3 overflow-x-auto">
          <div class="text-gray-400 text-xs mb-2">${lang || 'code'}</div>
          <pre><code class="text-green-400 text-sm">${escapeHtml(code.trim())}</code></pre>
        </div>`
      })
      
      // Handle inline code
      formatted = formatted.replace(/`([^`]+)`/g, '<code class="bg-gray-100 text-red-600 px-1.5 py-0.5 rounded text-sm">$1</code>')
      
      // Handle bold text
      formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-neuron-text-primary">$1</strong>')
      
      // Handle italic text
      formatted = formatted.replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
      
      // Handle headers
      formatted = formatted.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold text-neuron-text-primary mt-4 mb-2">$1</h3>')
      formatted = formatted.replace(/^## (.*$)/gm, '<h2 class="text-xl font-semibold text-neuron-text-primary mt-5 mb-3">$1</h2>')
      formatted = formatted.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold text-neuron-text-primary mt-6 mb-4">$1</h1>')
      
      // Handle unordered lists
      formatted = formatted.replace(/^\* (.*$)/gm, '<li class="ml-4 mb-1">• $1</li>')
      formatted = formatted.replace(/(<li.*<\/li>)/s, '<ul class="my-2 space-y-1">$1</ul>')
      
      // Handle numbered lists
      formatted = formatted.replace(/^\d+\. (.*$)/gm, '<li class="ml-4 mb-1 list-decimal">$1</li>')
      
      // Handle links (basic URL detection)
      formatted = formatted.replace(/(https?:\/\/[^\s<>"']+)/g, 
        '<a href="$1" target="_blank" class="text-accent-cyan hover:underline break-all">$1</a>')
      
      // Handle line breaks (convert double newlines to paragraphs)
      formatted = formatted.replace(/\n\n/g, '</p><p class="mb-3">')
      formatted = '<p class="mb-3">' + formatted + '</p>'
      
      // Handle single line breaks within paragraphs
      formatted = formatted.replace(/\n/g, '<br>')
      
      // Clean up empty paragraphs
      formatted = formatted.replace(/<p class="mb-3"><\/p>/g, '')
      
      return formatted
    }

    const escapeHtml = (text) => {
      const div = document.createElement('div')
      div.textContent = text
      return div.innerHTML
    }

    const sendMessage = async (content) => {
      if (!content || content.trim().length === 0) return

      // Add user message
      messages.value.push({
        role: 'user',
        content: content.trim(),
        timestamp: new Date()
      })

      // Clear input if from textarea
      if (content === userInput.value.trim()) {
        userInput.value = ''
      }

      isLoading.value = true
      isTyping.value = true

      try {
        const startTime = Date.now()

        // Create AI message placeholder for streaming
        const aiMessage = {
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          processingTime: null,
          isStreaming: true,
          suggestions: null
        }
        messages.value.push(aiMessage)

        await scrollToBottom()

        // Simple fetch for non-streaming response (you can adapt this to your API)
        const response = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { 
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            messages: messages.value
              .filter(m => m.role === "user")
              .map(m => m.content),
            stream: false
          })
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        
        // Hide typing indicator
        isTyping.value = false
        
        // Update message with formatted response
        aiMessage.content = data.response || data.text || data.message || 'No response received'
        aiMessage.processingTime = ((Date.now() - startTime) / 1000).toFixed(1) + "s"
        aiMessage.isStreaming = false
        aiMessage.suggestions = generateSmartSuggestions(content)
        
        await scrollToBottom()

      } catch (error) {
        console.error("Error:", error)
        isTyping.value = false
        
        // Update the last message with error
        const lastMessage = messages.value[messages.value.length - 1]
        if (lastMessage && lastMessage.role === 'assistant') {
          lastMessage.content = "I encountered an error while processing your request. Please try again."
          lastMessage.isStreaming = false
        }
      } finally {
        isLoading.value = false
        await scrollToBottom()
      }
    }

    const regenerateResponse = async (messageIndex) => {
      const userMessage = messages.value[messageIndex - 1]
      if (userMessage && userMessage.role === 'user') {
        // Remove the AI response and regenerate
        messages.value.splice(messageIndex, 1)
        await sendMessage(userMessage.content)
      }
    }

    const sendUserMessage = () => {
      if (canSend.value) {
        sendMessage(userInput.value)
      }
    }

    const copyMessage = async (content) => {
      try {
        // Strip HTML tags for plain text copy
        const plainText = content.replace(/<[^>]*>/g, '')
        await navigator.clipboard.writeText(plainText)
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
          return
        } else {
          event.preventDefault()
          sendUserMessage()
        }
      }
    }

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

      return ['Tell me more', 'Any sources?', 'Related topics?']
    }

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
      formatResponse,
      sendMessage,
      sendUserMessage,
      copyMessage,
      regenerateResponse,
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

/* Formatted response styles */
.formatted-response {
  line-height: 1.6;
}

.formatted-response p {
  margin-bottom: 0.75rem;
}

.formatted-response p:last-child {
  margin-bottom: 0;
}

.formatted-response h1,
.formatted-response h2,
.formatted-response h3 {
  font-weight: 600;
  line-height: 1.3;
}

.formatted-response ul,
.formatted-response ol {
  padding-left: 1rem;
  margin: 0.5rem 0;
}

.formatted-response li {
  margin-bottom: 0.25rem;
}

.formatted-response code {
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.875em;
}

.formatted-response .code-block {
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.875rem;
  line-height: 1.4;
}

.formatted-response .code-block pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

/* Streaming cursor animation */
@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}
</style>