<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center">
    <div class="max-w-md w-full text-center">
      <div class="mb-8">
        <div
          class="w-24 h-24 mx-auto mb-6 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center"
        >
          <ExclamationTriangleIcon class="w-12 h-12 text-white" />
        </div>

        <h1 class="text-6xl font-bold text-gray-900 mb-2">404</h1>
        <h2 class="text-2xl font-semibold text-gray-700 mb-4">
          Page Not Found
        </h2>
        <p class="text-gray-600 mb-8">
          Sorry, the page you're looking for doesn't exist or has been moved.
        </p>
      </div>

      <div class="space-y-4">
        <RouterLink to="/" class="btn-primary btn-lg w-full">
          <HomeIcon class="w-5 h-5" />
          Go Home
        </RouterLink>

        <button @click="goBack" class="btn-outline btn-lg w-full">
          <ArrowLeftIcon class="w-5 h-5" />
          Go Back
        </button>

        <RouterLink to="/search" class="btn-ghost btn-lg w-full">
          <MagnifyingGlassIcon class="w-5 h-5" />
          Search
        </RouterLink>
      </div>

      <div class="mt-8 text-sm text-gray-500">
        <p>If you believe this is an error, please contact support.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from "vue-router";
import {
  ExclamationTriangleIcon,
  HomeIcon,
  ArrowLeftIcon,
  MagnifyingGlassIcon,
} from "@heroicons/vue/24/outline";

const router = useRouter();

function goBack() {
  if (window.history.length > 1) {
    router.go(-1);
  } else {
    router.push("/");
  }
}
</script>
