<template>
  <footer class="bg-white border-t border-gray-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="grid md:grid-cols-4 gap-8">
        <!-- Brand Section -->
        <div class="md:col-span-1">
          <div class="flex items-center space-x-2 mb-4">
            <div
              class="w-6 h-6 bg-gradient-to-r from-primary-600 to-accent-600 rounded-md flex items-center justify-center"
            >
              <span class="text-white font-bold text-xs">NN</span>
            </div>
            <span class="font-bold text-gray-900">NewsNeuron</span>
          </div>
          <p class="text-gray-600 text-sm mb-4">
            Where News Meets Intelligence. AI-powered news analysis with hybrid
            vector-graph intelligence.
          </p>
          <div class="flex space-x-3">
            <a
              v-for="social in socialLinks"
              :key="social.name"
              :href="social.href"
              :title="social.name"
              class="text-gray-400 hover:text-gray-600 transition-colors duration-200"
              target="_blank"
              rel="noopener noreferrer"
            >
              <component :is="social.icon" class="w-5 h-5" />
            </a>
          </div>
        </div>

        <!-- Quick Links -->
        <div>
          <h3 class="font-semibold text-gray-900 mb-4">Features</h3>
          <ul class="space-y-2">
            <li v-for="link in featureLinks" :key="link.name">
              <RouterLink
                :to="link.to"
                class="text-gray-600 hover:text-gray-900 text-sm transition-colors duration-200"
              >
                {{ link.name }}
              </RouterLink>
            </li>
          </ul>
        </div>

        <!-- Resources -->
        <div>
          <h3 class="font-semibold text-gray-900 mb-4">Resources</h3>
          <ul class="space-y-2">
            <li v-for="link in resourceLinks" :key="link.name">
              <RouterLink
                :to="link.to"
                class="text-gray-600 hover:text-gray-900 text-sm transition-colors duration-200"
              >
                {{ link.name }}
              </RouterLink>
            </li>
          </ul>
        </div>

        <!-- Contact & Legal -->
        <div>
          <h3 class="font-semibold text-gray-900 mb-4">Company</h3>
          <ul class="space-y-2">
            <li v-for="link in companyLinks" :key="link.name">
              <RouterLink
                :to="link.to"
                class="text-gray-600 hover:text-gray-900 text-sm transition-colors duration-200"
              >
                {{ link.name }}
              </RouterLink>
            </li>
          </ul>
        </div>
      </div>

      <!-- Bottom Section -->
      <div class="mt-8 pt-8 border-t border-gray-200">
        <div class="md:flex md:items-center md:justify-between">
          <div class="text-sm text-gray-600">
            <p>
              &copy; {{ currentYear }} NewsNeuron Team. All rights reserved.
            </p>
            <p class="mt-1">
              Built with ❤️ using Vue.js, FastAPI, and AI technology.
            </p>
          </div>

          <div class="mt-4 md:mt-0 flex items-center space-x-4">
            <span class="text-xs text-gray-500">
              Version {{ appVersion }}
            </span>
            <div class="flex items-center space-x-1">
              <div
                class="w-2 h-2 rounded-full"
                :class="isOnline ? 'bg-success-400' : 'bg-error-400'"
              ></div>
              <span class="text-xs text-gray-500">
                {{ isOnline ? "Online" : "Offline" }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { computed, ref, onMounted, onUnmounted } from "vue";
import { RouterLink } from "vue-router";
import { useAppStore } from "@/stores/app";

// Icons (you can replace these with actual social media icons)
import {
  EnvelopeIcon,
  DocumentTextIcon,
  ShieldCheckIcon,
} from "@heroicons/vue/24/outline";

// Store
const appStore = useAppStore();

// State
const isOnline = ref(navigator.onLine);

// Computed
const currentYear = computed(() => new Date().getFullYear());
const appVersion = computed(() => appStore.appVersion);

// Links configuration
const featureLinks = [
  { name: "AI Chat", to: "/chat" },
  { name: "Flashcards", to: "/flashcards" },
  { name: "Timeline Analysis", to: "/timeline" },
  { name: "Search", to: "/search" },
  { name: "Dashboard", to: "/dashboard" },
];

const resourceLinks = [
  { name: "Documentation", to: "/docs" },
  { name: "API Reference", to: "/api-docs" },
  { name: "Tutorials", to: "/tutorials" },
  { name: "Blog", to: "/blog" },
  { name: "Help Center", to: "/help" },
];

const companyLinks = [
  { name: "About", to: "/about" },
  { name: "Contact", to: "/contact" },
  { name: "Privacy Policy", to: "/privacy" },
  { name: "Terms of Service", to: "/terms" },
  { name: "Status", to: "/status" },
];

const socialLinks = [
  {
    name: "GitHub",
    href: "https://github.com/newsneuron",
    icon: DocumentTextIcon, // Replace with actual GitHub icon
  },
  {
    name: "Twitter",
    href: "https://twitter.com/newsneuron",
    icon: EnvelopeIcon, // Replace with actual Twitter icon
  },
  {
    name: "LinkedIn",
    href: "https://linkedin.com/company/newsneuron",
    icon: ShieldCheckIcon, // Replace with actual LinkedIn icon
  },
  {
    name: "Email",
    href: "mailto:hello@newsneuron.com",
    icon: EnvelopeIcon,
  },
];

// Methods
function updateOnlineStatus() {
  isOnline.value = navigator.onLine;
}

// Lifecycle
onMounted(() => {
  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);
});

onUnmounted(() => {
  window.removeEventListener("online", updateOnlineStatus);
  window.removeEventListener("offline", updateOnlineStatus);
});
</script>
