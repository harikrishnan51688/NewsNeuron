# NewsNeuron 🧠
> **AI-Powered News Analysis Platform** - Think like a neural network, understand like a human

NewsNeuron is an advanced AI agent that redefines news analysis by thinking like a neural network to deliver precise, contextual insights. The system moves beyond standard search by combining the semantic power of Retrieval-Augmented Generation (RAG) with the deep relational understanding of a Knowledge Graph. By integrating vector-based search with relationship-driven graph traversal, NewsNeuron processes information like interconnected neurons, understanding not just the content of the news, but the complex relationships between people, organizations, and events.

## 📹 Presentation Video
https://drive.google.com/file/d/1NY46p2MO2ReZBgco64u9mAvJsqJCvxi1/view?usp=sharing

## 🏗️ System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        NewsNeuron Platform                      │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Vue.js)                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Home View     │  │   Chat View     │  │ Flashcards View │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │         │
│           └─────────────────────┼─────────────────────┘         │
│                                 │                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              API Gateway (FastAPI)                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend Services                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   News Agent    │  │  Retrieval      │  │  Knowledge      │ │
│  │   (LangChain)   │  │  System         │  │  Graph          │ │
│  │                 │  │  (RAG)          │  │  (Neo4j)        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   PostgreSQL    │  │    Pinecone     │  │     Neo4j       │ │
│  │   (Metadata)    │  │   (Vectors)     │  │ (Relationships) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 🎨 Frontend Layer
- **Vue.js 3 Application**: Modern reactive frontend with component-based architecture
- **Views**: Home, Chat, Flashcards, Search, Timeline, About
- **State Management**: Pinia for centralized state management
- **Styling**: Tailwind CSS for responsive design
- **Build Tool**: Vite for fast development and optimized builds

#### 🔌 API Layer
- **FastAPI Backend**: RESTful API with automatic OpenAPI documentation
- **Route Handlers**: Modular route organization for different functionalities
- **Middleware**: CORS, authentication, and error handling
- **Real-time Communication**: WebSocket support for live updates

#### 🤖 AI Processing Layer
- **News Agent**: LangChain-based agent for intelligent news processing
- **Retrieval System**: RAG implementation with semantic search
- **Knowledge Graph**: Neo4j-based relationship mapping
- **Embeddings**: Vector representations for semantic understanding

#### 💾 Data Storage Layer
- **PostgreSQL**: Primary database for metadata, user data, and structured information
- **Pinecone**: Vector database for semantic search and similarity matching
- **Neo4j**: Graph database for knowledge relationships and entity connections

### Data Flow
1. **News Ingestion**: External news sources → News Agent → Data Processing
2. **Content Analysis**: Raw news → Embeddings → Vector Storage (Pinecone)
3. **Relationship Mapping**: Entities → Knowledge Graph → Neo4j
4. **User Queries**: Frontend → API → RAG System → Response Generation
5. **Knowledge Retrieval**: Query → Vector Search + Graph Traversal → Contextual Results

## 🛠️ Tech Stack

### Frontend
- **Vue.js 3** - Progressive JavaScript framework
- **Tailwind CSS** - Utility-first CSS framework
- **Vite** - Fast build tool and dev server
- **Pinia** - State management

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.12** - Language
- **LangChain** - LLM application framework
- **OpenAI/OpenRouter** - Large language models
- **Pinecone** - Vector database for embeddings

### Databases
- **PostgreSQL** - Primary relational database
- **Neo4j** - Knowledge graph database
- **Pinecone** - Vector similarity search

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Web server and reverse proxy

## 🚀 Features

- **🧠 Neural Network Thinking**: AI processes news like interconnected neurons
- **🔍 Semantic Search**: Advanced RAG system for contextual understanding
- **📊 Knowledge Graph**: Relationship mapping between entities and events
- **💬 Interactive Chat**: Natural language interface for news queries
- **📚 Flashcard Generation**: AI-powered learning materials from news content
- **📈 Timeline Visualization**: Chronological view of news events
- **🎯 Personalized Insights**: Tailored news analysis based on user preferences

## 🏭 Installation & Setup

### Prerequisites

- **Python 3.12+** installed on your system
- **Node.js 18+** for frontend development
- **uv** for Python dependency management ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker** (optional, for database containers)

### Quick Start

#### 1. Clone the Repository
```bash
git clone https://github.com/harikrishnan51688/NewsNeuron
cd NewsNeuron
```

#### 2. Install Python Dependencies
```bash
uv sync
```

#### 3. Set Up Databases

**Option A: Using Docker (Recommended)**
```bash
# PostgreSQL
docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres

# Neo4j
docker run --name neo4j -e NEO4J_AUTH=neo4j/password -p 7687:7687 -p 7474:7474 -d neo4j
```

**Option B: Local Installation**
- Install PostgreSQL locally and create a database named `newsneuron`
- Install Neo4j locally with default credentials

#### 4. Environment Configuration

Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Fill in the required environment variables:
```env
# API Keys
OPENROUTER_API_KEY="your_openrouter_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_ENVIRONMENT="us-east-1"
PINECONE_INDEX_NAME="news-articles"

# Database URLs
POSTGRES_URL="postgresql://postgres:password@localhost:5432/newsneuron"
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password"
```

#### 5. Run the Application

**Backend (API Server)**
```bash
uv run uvicorn api.routes:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (Development Server)**
```bash
cd frontend
npm install
npm run dev
```

The application will be available at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🐳 Docker Deployment

### Development Environment
```bash
docker-compose up -d
```

### Production Environment
```bash
docker-compose -f docker-compose.prod.yml up -d
```



