# NewsNeuron Frontend

A modern Vue.js 3 application built with Vite, providing an intelligent interface for AI-powered news analysis.

[![Vue 3.3+](https://img.shields.io/badge/Vue-3.3+-green.svg)](https://vuejs.org/)
[![Vite 7+](https://img.shields.io/badge/Vite-7+-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4+-38B2AC.svg)](https://tailwindcss.com/)
[![ESLint 9+](https://img.shields.io/badge/ESLint-9+-4B32C3.svg)](https://eslint.org/)

## 🚀 Features

- **Modern Vue.js 3**: Composition API, `<script setup>`, and latest best practices
- **Lightning Fast**: Powered by Vite for instant development and optimized production builds
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **Real-time Interface**: Reactive UI for chat, search, and timeline interactions
- **Component Library**: Reusable UI components with consistent design system
- **State Management**: Pinia for lightweight and efficient state management
- **Type Safety**: ESLint 9 with modern flat config for code quality
- **Testing Ready**: Vitest for unit and integration testing

## 📁 Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/        # Reusable Vue components
│   │   ├── layout/       # Layout components (Header, Footer, Navigation)
│   │   ├── ui/           # Generic UI components (Button, Modal, Card)
│   │   ├── flashcards/   # Flashcard-specific components
│   │   └── charts/       # Chart and visualization components
│   ├── views/            # Page-level components (Router views)
│   ├── stores/           # Pinia stores for state management
│   ├── services/         # API services and HTTP clients
│   ├── utils/            # Utility functions and helpers
│   ├── router/           # Vue Router configuration
│   ├── assets/           # Static assets (images, fonts)
│   ├── style.css         # Global styles and Tailwind config
│   └── main.js           # Application entry point
├── eslint.config.js      # ESLint 9 flat configuration
├── tailwind.config.js    # Tailwind CSS configuration
├── vite.config.js        # Vite build configuration
└── package.json          # Dependencies and scripts
```

## 🛠️ Development Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **npm** or **yarn**

### Installation

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   VITE_API_VERSION=v1
   VITE_APP_NAME=NewsNeuron
   VITE_ENABLE_DEBUG=true
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

The application will be available at `http://localhost:5173`

## 📜 Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server with HMR |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run test` | Run unit tests with Vitest |
| `npm run test:ui` | Run tests with UI dashboard |
| `npm run lint` | Lint code with ESLint 9 |
| `npm run format` | Format code with Prettier |

## 🎨 UI Components

### Layout Components

- **AppNavigation**: Main navigation with responsive menu
- **AppFooter**: Site footer with links and branding
- **Sidebar**: Collapsible sidebar for additional navigation

### UI Components

- **Button**: Various button styles and states
- **Card**: Flexible card container with header/body/footer
- **Modal**: Accessible modal dialogs
- **LoadingIndicator**: Global loading states
- **ToastNotification**: Toast messages for user feedback

### Feature Components

- **FlashcardPreview**: News flashcard display
- **ChatInterface**: AI chat conversation UI
- **SearchResults**: Search result display with filters
- **TimelineView**: Interactive timeline visualization

## 🏪 State Management

We use **Pinia** for state management with the following stores:

### App Store (`stores/app.js`)
- Global application state
- Theme management (light/dark mode)
- Loading states and notifications
- Error handling

### Flashcards Store (`stores/flashcards.js`)
- Flashcard data management
- Filtering and search
- CRUD operations

### Chat Store (`stores/chat.js`)
- Conversation management
- Message history
- Real-time updates

### Example Store Usage

```javascript
import { useAppStore } from '@/stores/app'
import { useFlashcardsStore } from '@/stores/flashcards'

export default {
  setup() {
    const appStore = useAppStore()
    const flashcardsStore = useFlashcardsStore()

    // Access reactive state
    const isLoading = computed(() => appStore.isLoading)
    
    // Call actions
    const loadFlashcards = () => flashcardsStore.getFlashcards()

    return { isLoading, loadFlashcards }
  }
}
```

## 🌐 API Integration

### HTTP Client

We use **Axios** with custom interceptors for API communication:

```javascript
// services/api.js
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 30000
})
```

### Service Layer

API calls are organized by feature:

```javascript
// Chat API
export const chatAPI = {
  sendMessage: (data) => api.post('/chat/', data),
  getConversations: (params) => api.get('/chat/conversations', { params }),
  getHistory: (id, params) => api.get(`/chat/conversations/${id}/history`, { params })
}

// Flashcards API
export const flashcardsAPI = {
  getFlashcards: (params) => api.get('/flashcards/', { params }),
  generateFlashcards: (data) => api.post('/flashcards/', data),
  getTrendingTopics: (params) => api.get('/flashcards/topics/trending', { params })
}
```

## 🎨 Styling

### Tailwind CSS

We use **Tailwind CSS** for styling with custom configuration:

```javascript
// tailwind.config.js
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: { /* custom color palette */ },
        accent: { /* accent colors */ }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace']
      }
    }
  }
}
```

### Component Classes

Custom component classes in `style.css`:

```css
/* Button components */
.btn {
  @apply inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm transition-all duration-200;
}

.btn-primary {
  @apply btn bg-primary-600 text-white hover:bg-primary-700;
}

/* Card components */
.card {
  @apply bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden;
}
```

## 🧪 Testing

### Unit Testing with Vitest

```javascript
// Component test example
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import FlashcardPreview from '@/components/flashcards/FlashcardPreview.vue'

describe('FlashcardPreview', () => {
  it('renders flashcard data correctly', () => {
    const flashcard = {
      title: 'Test Title',
      summary: 'Test Summary',
      key_points: ['Point 1', 'Point 2']
    }

    const wrapper = mount(FlashcardPreview, {
      props: { flashcard }
    })

    expect(wrapper.text()).toContain('Test Title')
    expect(wrapper.text()).toContain('Test Summary')
  })
})
```

### Running Tests

```bash
# Run all tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

## 🔧 Code Quality

### ESLint 9 Configuration

We use ESLint 9 with flat config format:

```javascript
// eslint.config.js
import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import prettierConfig from '@vue/eslint-config-prettier'

export default [
  {
    ignores: ['dist/**', 'node_modules/**']
  },
  js.configs.recommended,
  ...pluginVue.configs['flat/essential'],
  prettierConfig,
  {
    rules: {
      'vue/multi-word-component-names': 'off',
      'no-unused-vars': 'warn'
    }
  }
]
```

### Code Formatting

- **ESLint**: Linting and code quality rules
- **Prettier**: Code formatting (integrated with ESLint)
- **Vue Style Guide**: Following official Vue.js conventions

## 🚀 Build & Deployment

### Production Build

```bash
npm run build
```

This creates an optimized build in the `dist/` directory with:
- **Code splitting**: Automatic vendor and component chunking
- **Asset optimization**: Minified CSS/JS and optimized images
- **Tree shaking**: Dead code elimination
- **Source maps**: For debugging production issues

### Build Configuration

```javascript
// vite.config.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
          charts: ['chart.js', 'vue-chartjs', 'd3']
        }
      }
    }
  }
})
```

### Deployment

The frontend is optimized for **Vercel** deployment:

1. **Automatic deployment** on push to main branch
2. **Preview deployments** for pull requests
3. **Environment variables** configured in Vercel dashboard
4. **Custom domain** support with SSL

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_API_VERSION` | API version | `v1` |
| `VITE_APP_NAME` | Application name | `NewsNeuron` |
| `VITE_ENABLE_DEBUG` | Enable debug mode | `false` |
| `VITE_SENTRY_DSN` | Sentry error tracking | - |

### Vite Configuration

Key configuration options:

```javascript
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Kill process on port 5173
   lsof -ti:5173 | xargs kill -9
   ```

2. **ESLint errors after update**:
   ```bash
   # Clear ESLint cache
   npm run lint -- --cache-clear
   ```

3. **Dependency conflicts**:
   ```bash
   # Clean install
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Build fails**:
   ```bash
   # Check for type errors
   npm run lint
   # Clean build directory
   rm -rf dist && npm run build
   ```

## 📚 Resources

### Vue.js Resources
- [Vue.js 3 Documentation](https://vuejs.org/)
- [Vue Composition API](https://vuejs.org/guide/extras/composition-api-faq.html)
- [Vue Style Guide](https://vuejs.org/style-guide/)

### Development Tools
- [Vite Documentation](https://vitejs.dev/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Pinia Documentation](https://pinia.vuejs.org/)
- [Vitest Documentation](https://vitest.dev/)

### Code Quality
- [ESLint Rules](https://eslint.org/docs/rules/)
- [Vue ESLint Plugin](https://eslint.vuejs.org/)
- [Prettier Configuration](https://prettier.io/docs/en/configuration.html)

## 🤝 Contributing

1. Follow the [Vue.js Style Guide](https://vuejs.org/style-guide/)
2. Use `<script setup>` syntax for new components
3. Write tests for new features
4. Update documentation for changes
5. Run linting before commits: `npm run lint`

For detailed contributing guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
