import { createRouter, createWebHistory } from "vue-router";
import { useAppStore } from "@/stores/app";

// Views
import HomeView from "@/views/HomeView.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: HomeView,
      meta: {
        title: "NewsNeuron - Where News Meets Intelligence",
        description:
          "AI-powered news analysis with hybrid vector-graph intelligence",
      },
    },
    {
      path: "/chat",
      name: "chat",
      component: () => import("@/views/ChatView.vue"),
      meta: {
        title: "AI Chat - NewsNeuron",
        description: "Chat with our AI assistant about news and current events",
      },
    },
    {
      path: "/flashcards",
      name: "flashcards",
      component: () => import("@/views/FlashcardsView.vue"),
      meta: {
        title: "News Flashcards - NewsNeuron",
        description: "Quick news summaries in flashcard format",
      },
    },
    {
      path: "/timeline/:entity?",
      name: "timeline",
      component: () => import("@/views/TimelineView.vue"),
      meta: {
        title: "Timeline Analysis - NewsNeuron",
        description: "Visualize story evolution and entity timelines",
      },
    },
    {
      path: "/search",
      name: "search",
      component: () => import("@/views/SearchView.vue"),
      meta: {
        title: "Search News - NewsNeuron",
        description: "Advanced semantic search through news articles",
      },
    },
    // Catch-all 404 route
    {
      path: "/:pathMatch(.*)*",
      name: "not-found",
      component: () => import("@/views/NotFoundView.vue"),
      meta: {
        title: "Page Not Found - NewsNeuron",
        description: "The page you are looking for does not exist",
      },
    },
  ],
  scrollBehavior(to, from, savedPosition) {
    // Always scroll to top on route change unless user has saved position
    if (savedPosition) {
      return savedPosition;
    } else {
      return { top: 0 };
    }
  },
});

// Global navigation guards
router.beforeEach(async (to, from, next) => {
  const appStore = useAppStore();

  console.log('Router beforeEach:', { from: from.path, to: to.path, name: to.name });

  try {
    // Set loading state
    appStore.setLoading(true);

    // Update page title and meta
    if (to.meta?.title) {
      document.title = to.meta.title;
    }

    if (to.meta?.description) {
      const metaDescription = document.querySelector('meta[name="description"]');
      if (metaDescription) {
        metaDescription.setAttribute("content", to.meta.description);
      }
    }

    console.log('Router beforeEach: calling next()');
    next();
  } catch (error) {
    console.error("Router beforeEach error:", error);
    appStore.setLoading(false);
    next();
  }
});

router.afterEach((to, from, failure) => {
  const appStore = useAppStore();

  console.log('Router afterEach:', { from: from.path, to: to.path, failure });

  try {
    // Clear loading state regardless of success/failure
    appStore.setLoading(false);

    // Log route changes for debugging
    if (failure) {
      console.error("Route navigation failed:", failure);
    } else {
      console.log("Route navigation successful:", to.path);
    }

    // Analytics tracking (when implemented)
    if (import.meta.env.VITE_GOOGLE_ANALYTICS_ID) {
      // Track page view
      console.log("Track page view:", to.path);
    }
  } catch (error) {
    console.error("Router afterEach error:", error);
    // Ensure loading is cleared even if there's an error
    appStore.setLoading(false);
  }
});

// Handle route errors
router.onError((error) => {
  console.error("Router error:", error);
  const appStore = useAppStore();
  appStore.setLoading(false);
  appStore.setError("Failed to load page. Please try again.");
});

export default router;
