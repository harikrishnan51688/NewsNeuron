<template>
  <div class="chat-view min-h-screen bg-neuron-bg-primary">

    <!-- Subtle Background Particles -->
    <div class="fixed inset-0 overflow-hidden pointer-events-none">
      <div v-for="i in 8" :key="i" :class="getParticleColor(i)" class="absolute w-0.5 h-0.5 rounded-full animate-pulse"
        :style="particleStyle(i)">
      </div>
    </div>

    <!-- Main Chat Interface - Simplified -->
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

          <!-- Quick Suggestions - Smaller and closer -->
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
            <div class="max-w-xs sm:max-w-md lg:max-w-lg">
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
            <div class="max-w-xs sm:max-w-md lg:max-w-2xl">
              <div class="bg-neuron-bg-content border border-neuron-border p-4 rounded-2xl rounded-tl-sm shadow-sm">
                <!-- Formatted message content -->
                <div class="formatted-message text-sm leading-relaxed text-neuron-text-primary" 
                     v-html="formatMessage(message.content)">
                </div>
                <span v-if="message.isStreaming" class="inline-block w-2 h-4 bg-accent-emerald animate-pulse ml-1"></span>

                <!-- Processing time (subtle) -->
                <div v-if="message.processingTime" class="text-xs text-neuron-text-secondary/50 mt-1">
                  {{ message.processingTime }}
                </div>

                <!-- Copy Button -->
                <button @click="copyMessage(message.content)"
                  class="mt-2 p-1 hover:bg-neuron-bg-primary rounded transition-colors" title="Copy message">
                  <Copy class="w-3 h-3 text-neuron-text-secondary" />
                </button>
              </div>

              <!-- Suggested Follow-ups (for latest AI message) -->
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

      <!-- Input Area - Streamlined -->
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

    // Enhanced message formatting function
    const formatMessage = (content) => {
      if (!content) return ''
      
      let formatted = content
      
      // Convert markdown-style formatting
      formatted = formatted
        // Bold text
        .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-neuron-text-primary">$1</strong>')
        // Italic text
        .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
        // Code blocks (triple backticks)
        .replace(/```([\s\S]*?)```/g, '<pre class="bg-neuron-bg-primary/50 border border-neuron-border rounded-lg p-3 my-2 text-sm font-mono overflow-x-auto"><code>$1</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code class="bg-neuron-bg-primary/50 px-1.5 py-0.5 rounded text-sm font-mono">$1</code>')
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-accent-cyan hover:text-accent-emerald transition-colors underline">$1</a>')
        // Citations (antml:cite tags)
        .replace(/]*>(.*?)<\/antml:cite>/g, '<span class="bg-accent-emerald/10 text-accent-emerald px-1 py-0.5 rounded text-sm border-l-2 border-accent-emerald/30">$1</span>')
      
      // Handle lists
      const lines = formatted.split('\n')
      const processedLines = []
      let inList = false
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i]
        
        // Detect bullet points
        if (line.match(/^[\s]*[-•*]\s+/)) {
          if (!inList) {
            processedLines.push('<ul class="list-disc pl-6 space-y-1 my-2">')
            inList = true
          }
          const content = line.replace(/^[\s]*[-•*]\s+/, '')
          processedLines.push(`<li class="text-neuron-text-primary">${content}</li>`)
        }
        // Detect numbered lists
        else if (line.match(/^[\s]*\d+\.\s+/)) {
          if (!inList) {
            processedLines.push('<ol class="list-decimal pl-6 space-y-1 my-2">')
            inList = true
          }
          const content = line.replace(/^[\s]*\d+\.\s+/, '')
          processedLines.push(`<li class="text-neuron-text-primary">${content}</li>`)
        }
        // Regular line
        else {
          if (inList) {
            processedLines.push('</ul>')
            inList = false
          }
          
          // Handle headers
          if (line.startsWith('# ')) {
            processedLines.push(`<h1 class="text-lg font-semibold text-neuron-text-primary mt-4 mb-2">${line.substring(2)}</h1>`)
          } else if (line.startsWith('## ')) {
            processedLines.push(`<h2 class="text-base font-semibold text-neuron-text-primary mt-3 mb-2">${line.substring(3)}</h2>`)
          } else if (line.startsWith('### ')) {
            processedLines.push(`<h3 class="text-sm font-semibold text-neuron-text-primary mt-2 mb-1">${line.substring(4)}</h3>`)
          } else if (line.trim() === '') {
            processedLines.push('<br>')
          } else {
            processedLines.push(`<p class="mb-2">${line}</p>`)
          }
        }
      }
      
      // Close any open list
      if (inList) {
        processedLines.push('</ul>')
      }
      
      return processedLines.join('')
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

        // Scroll to show the new message
        await scrollToBottom()

        // Call backend with streaming
        const response = await fetch("http://10.20.4.2:8000/chat", {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
          },
          body: JSON.stringify({
            messages: messages.value
              .filter(m => m.role === "user")
              .map(m => m.content),
            stream: true
          })
        })

        console.log("Response status:", response.status)
        console.log("Response headers:", Object.fromEntries(response.headers.entries()))

        if (!response.ok) {
          const errorText = await response.text()
          console.error("API Error:", errorText)
          throw new Error(`HTTP error! status: ${response.status} - ${errorText}`)
        }

        if (!response.body) {
          throw new Error("No response body")
        }

        // Check if it's actually streaming
        const contentType = response.headers.get('content-type')
        console.log("Content-Type:", contentType)

        // Hide typing indicator once streaming starts
        isTyping.value = false

        const reader = response.body.getReader()
        const decoder = new TextDecoder("utf-8")
        let buffer = ""

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              console.log("Stream completed")
              break
            }

            // Add new data to buffer
            const chunk = decoder.decode(value, { stream: true })
            console.log("Received chunk:", chunk)
            buffer += chunk
            
            // Process complete lines (handle both \n\n and \n as separators)
            const lines = buffer.split(/\n+/)
            buffer = lines.pop() || "" // Keep incomplete line in buffer

            for (const line of lines) {
              if (line.trim() === "") continue
              
              console.log("Processing line:", line)
              
              if (line.startsWith("data: ")) {
                try {
                  const jsonData = line.slice(6).trim()
                  if (jsonData === "") continue
                  
                  console.log("Parsing JSON:", jsonData)
                  const data = JSON.parse(jsonData)

                  if (data.type === "content") {
                    aiMessage.content += data.data
                    console.log("Added content:", data.data)
                    // Scroll to bottom as content streams in
                    await scrollToBottom()
                  }
                  else if (data.type === "done") {
                    aiMessage.processingTime = ((Date.now() - startTime) / 1000).toFixed(1) + "s"
                    aiMessage.isStreaming = false
                    
                    // Generate smart suggestions based on the conversation
                    aiMessage.suggestions = generateSmartSuggestions(content)
                    
                    console.log("Stream done")
                    await scrollToBottom()
                    break
                  }
                  else if (data.type === "error") {
                    aiMessage.content = "⚠️ Error: " + data.data
                    aiMessage.isStreaming = false
                    console.error("Stream error:", data.data)
                    break
                  }
                } catch (error) {
                  console.error("Error parsing JSON:", error, "Line:", line)
                  continue
                }
              } else {
                // Handle non-SSE format - maybe it's just JSON chunks
                try {
                  const data = JSON.parse(line)
                  if (data.type === "content") {
                    aiMessage.content += data.data
                    await scrollToBottom()
                  } else if (data.type === "done") {
                    aiMessage.processingTime = ((Date.now() - startTime) / 1000).toFixed(1) + "s"
                    aiMessage.isStreaming = false
                    aiMessage.suggestions = generateSmartSuggestions(content)
                    await scrollToBottom()
                    break
                  }
                } catch (e) {
                  console.log("Not JSON, skipping line:", line)
                }
              }
            }
          }
        } catch (streamError) {
          console.error("Stream reading error:", streamError)
          throw streamError
        }

      } catch (error) {
        console.error("Streaming error:", error)
        
        // Update the last message with error
        const lastMessage = messages.value[messages.value.length - 1]
        if (lastMessage && lastMessage.role === 'assistant') {
          lastMessage.content = "I encountered an error while processing your request. Please try again."
          lastMessage.isStreaming = false
        } else {
          messages.value.push({
            role: "assistant",
            content: "I encountered an error while processing your request. Please try again.",
            timestamp: new Date(),
            isStreaming: false
          })
        }
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
        // Optional: Show a brief success indicator
      } catch (error) {
        console.error('Failed to copy message:', error)
        // Fallback for older browsers
        const textArea = document.createElement('textarea')
        textArea.value = content
        document.body.appendChild(textArea)
        textArea.select()
        try {
          document.execCommand('copy')
        } catch (fallbackError) {
          console.error('Fallback copy failed:', fallbackError)
        }
        document.body.removeChild(textArea)
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
      formatMessage,
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

/* Streaming cursor animation */
@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* Formatted message styles */
.formatted-message {
  line-height: 1.6;
}

.formatted-message h1,
.formatted-message h2,
.formatted-message h3 {
  margin-top: 1rem;
  margin-bottom: 0.5rem;
}

.formatted-message p {
  margin-bottom: 0.75rem;
}

.formatted-message p:last-child {
  margin-bottom: 0;
}

.formatted-message ul,
.formatted-message ol {
  margin: 0.5rem 0;
}

.formatted-message pre {
  white-space: pre-wrap;
  word-wrap: break-word;
}

.formatted-message code {
  word-wrap: break-word;
}

.formatted-message a {
  word-wrap: break-word;
}

/* Citation styling */
.formatted-message .citation {
  background: rgba(16, 185, 129, 0.1);
  color: rgb(16, 185, 129);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.875rem;
  border-left: 2px solid rgba(16, 185, 129, 0.3);
}
</style>