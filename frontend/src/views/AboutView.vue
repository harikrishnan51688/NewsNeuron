<template>
  <div class="min-h-screen bg-neuron-bg-primary">
    <div class="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div class="text-center mb-12">
        <h1 class="text-4xl text-display text-gradient mb-4">About NewsNeuron</h1>
        <p class="text-xl text-elegant text-neuron-text-secondary">Where News Meets Intelligence</p>
      </div>

      <div class="bg-neuron-bg-content rounded-lg shadow-neuron-lg border border-neuron-border p-8 mb-8">
        <h2 class="text-2xl text-heading text-neuron-text-primary mb-4">Our Mission</h2>
        <p class="text-body-sans text-neuron-text-secondary mb-6">
          NewsNeuron is designed to transform how we understand and interact
          with news through AI-powered intelligence. We combine cutting-edge
          vector search with knowledge graph technology to deliver contextual
          insights that matter.
        </p>

        <h2 class="text-2xl text-heading text-neuron-text-primary mb-4">Technology Stack</h2>
        <div class="grid md:grid-cols-2 gap-6 mb-6">
          <div>
            <h3 class="text-heading text-neuron-text-primary mb-2">Frontend</h3>
            <ul class="text-body-sans text-neuron-text-secondary space-y-1">
              <li>• Vue.js 3 with Composition API</li>
              <li>• Tailwind CSS for styling</li>
              <li>• Pinia for state management</li>
              <li>• Vite for build tooling</li>
            </ul>
          </div>

          <div>
            <h3 class="text-heading text-neuron-text-primary mb-2">Backend</h3>
            <ul class="text-body-sans text-neuron-text-secondary space-y-1">
              <li>• FastAPI with Python</li>
              <li>• Supabase (PostgreSQL + pgvector)</li>
              <li>• Neo4j for knowledge graphs</li>
              <li>• OpenAI for embeddings & LLM</li>
            </ul>
          </div>
        </div>

        <h2 class="text-2xl text-heading text-neuron-text-primary mb-4">Team</h2>
        <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div
            v-for="member in teamMembers"
            :key="member.name"
            class="text-center"
          >
            <div
              class="w-16 h-16 bg-gradient-to-r from-neuron-glow to-neuron-glow-hover rounded-full mx-auto mb-3 flex items-center justify-center"
            >
              <span class="text-white font-semibold text-lg">{{
                member.initials
              }}</span>
            </div>
            <h3 class="font-semibold text-heading text-neuron-text-primary">{{ member.name }}</h3>
            <p class="text-body-serif text-neuron-text-secondary text-sm">{{ member.role }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const teamMembers = [
  { name: "HARI", role: "Project Lead & Backend Architecture", initials: "H" },
  { name: "VARADA", role: "Frontend Development & UI/UX", initials: "V" },
  { name: "SANJAI", role: "AI Agent & LangGraph Integration", initials: "S" },
  { name: "ARYA", role: "Data Processing & Database Design", initials: "A" },
  { name: "KOHIL", role: "DevOps & Deployment", initials: "K" },
];
</script>
