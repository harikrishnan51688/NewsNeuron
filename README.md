## NewsNeuron 🧠
> **AI-Powered News Analysis Platform** - Think like a neural network, understand like a human

NewsNeuron is an advanced AI agent that redefines news analysis by thinking like a neural network to deliver precise, contextual insights. The system moves beyond standard search by combining the semantic power of Retrieval-Augmented Generation (RAG) with the deep relational understanding of a Knowledge Graph. By integrating vector-based search with relationship-driven graph traversal, NewsNeuron processes information like interconnected neurons, understanding not just the content of the news, but the complex relationships between people, organizations, and events.

## Presentation Video
https://drive.google.com/file/d/1NY46p2MO2ReZBgco64u9mAvJsqJCvxi1/view?usp=sharing

## System Architecture Overview
<img src="docs/image.png" alt="drawing" width="400"/>

## 🛠️ Tech Stack

### Frontend
- **Vue.js 3** - Progressive JavaScript framework
- **Tailwind CSS** - Utility-first CSS framework
- **Vite** - Fast build tool and dev server
- **Pinia** - State management
- **Chart.js** - Data visualization

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.12** - Programming language
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


## 🏭 Project Setup Guide

This project uses uv for Python dependency management and virtual environment handling.

### Prerequisites

- Python 3.12+ installed on your system
- uv installed (installation guide)

## Quick Start
1. Clone the Repository
```bash
git clone https://github.com/harikrishnan51688/NewsNeuron
cd NewsNeuron
```
2. Install Dependencies
```bash
uv sync
```
- For postgres, you can use postgres docker image:
```bash
docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres
```
or install postgres locally and create a database named `newsneuron`.

- For Neo4j, you can use the Neo4j docker image:
```bash
docker run --name neo4j -e NEO4J_AUTH=neo4j/password -p 7687:7687 -p 7474:7474 -d neo4j
```
or install Neo4j locally and set up a database with the default credentials.

3. Set Up Environment Variables

Create a `.env` file in the root directory and add the following variables:
You can copy .env.example to .env and fill in the required values.
```env
OPENROUTER_API_KEY="your_key"
POSTGRES_URL="your_postgres_url"
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_ENVIRONMENT="us-east-1"
PINECONE_INDEX_NAME="news-articles"
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password"
```
To run api
```bash
uv run uvicorn api.routes:app --reload
```

### To run Frontend
```bash
cd frontend
```
Install dependencies
```bash
npm install
```
Run the server
```bash
npm run dev
```
