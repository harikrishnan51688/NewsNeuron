/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,js,ts,jsx,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      // Synaptic Night Color Palette - Effortless Clarity Design System
      colors: {
        // Core Brand Colors
        'neuron': {
          'bg-primary': '#0D1117',     // Deep charcoal/navy background
          'bg-content': '#161B22',     // Content containers
          'border': '#30363D',         // Borders & dividers
          'text-primary': '#E6EDF3',   // Primary text
          'text-secondary': '#8B949E', // Secondary text/metadata
          'glow': '#00A7E1',          // Neuron Glow accent
          'glow-hover': '#33BBF3',     // Accent hover state
        },
        
        // Extended semantic colors
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe', 
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#00A7E1', // Neuron Glow
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        
        // Dark theme color aliases
        'dark-background': '#0D1117',     // Neuron bg-primary
        'dark-card': '#161B22',           // Neuron bg-content  
        'dark-border': '#30363D',         // Neuron border
        'text-primary': '#E6EDF3',        // Neuron text-primary
        'text-secondary': '#8B949E',      // Neuron text-secondary
        'text-muted': '#6E7681',          // Dimmer text
        'neuron-glow': '#00A7E1',         // Accent color
        
        // Logo Colors - Unique Purple Palette
        'logo-primary': '#8B5CF6',       // Purple
        'logo-secondary': '#A855F7',     // Lighter purple
        'logo-glow': '#C084FC',          // Soft purple glow
        
        // Accent Colors - More Variety
        'accent-emerald': '#10B981',     // Green accent
        'accent-amber': '#F59E0B',       // Yellow accent  
        'accent-rose': '#F43F5E',        // Pink accent
        'accent-violet': '#8B5CF6',      // Purple accent
        'accent-cyan': '#06B6D4',        // Cyan accent
        'accent-orange': '#EA580C',      // Orange accent
        
        // Grayscale system optimized for dark mode
        gray: {
          50: '#E6EDF3',   // Primary text
          100: '#D0D7DE',  
          200: '#AFBAC5',  
          300: '#8B949E',  // Secondary text
          400: '#6E7681',  
          500: '#57606A',  
          600: '#424A53',  
          700: '#32383F',  
          800: '#24292F',  
          850: '#1C2128',  
          900: '#161B22',  // Content background
          925: '#0F1419',  
          950: '#0D1117',  // Primary background
        },
        
        // Success/Error states with dark mode optimization
        success: {
          50: '#f0fdf4',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
        },
        
        warning: {
          50: '#fffbeb', 
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
        },
        
        error: {
          50: '#fef2f2',
          500: '#ef4444', 
          600: '#dc2626',
          700: '#b91c1c',
        },
      },
      
      // Typography System - Ultra-Modern Design
      fontFamily: {
        'display': ['Space Grotesk', 'system-ui', 'sans-serif'],           // Logo & Display text
        'heading': ['Space Grotesk', 'system-ui', 'sans-serif'],           // Headings (lighter)
        'body': ['Space Grotesk', 'system-ui', 'sans-serif'],              // Body text  
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],              // Code/Technical
        'serif': ['Playfair Display', 'Georgia', 'serif'],                 // Elegant accents
      },
      
      fontSize: {
        // 8pt grid system typography scale
        'xs': ['12px', { lineHeight: '16px', letterSpacing: '0.01em' }],    // Label
        'sm': ['14px', { lineHeight: '20px', letterSpacing: '0.01em' }],    // Small/Metadata  
        'base': ['16px', { lineHeight: '24px', letterSpacing: '0' }],       // Body/Default
        'lg': ['18px', { lineHeight: '28px', letterSpacing: '-0.01em' }],   
        'xl': ['20px', { lineHeight: '28px', letterSpacing: '-0.01em' }],   // Card Title
        '2xl': ['24px', { lineHeight: '32px', letterSpacing: '-0.02em' }],  // H3
        '3xl': ['32px', { lineHeight: '40px', letterSpacing: '-0.02em' }],  // H2
        '4xl': ['48px', { lineHeight: '56px', letterSpacing: '-0.03em' }],  // H1
      },
      
      fontWeight: {
        'normal': '400',
        'medium': '500', 
        'bold': '700',
      },
      
      // 8pt Grid System Spacing
      spacing: {
        '0.5': '2px',   // 0.25 * 8
        '1': '4px',     // 0.5 * 8  
        '1.5': '6px',   // 0.75 * 8
        '2': '8px',     // 1 * 8
        '3': '12px',    // 1.5 * 8
        '4': '16px',    // 2 * 8
        '5': '20px',    // 2.5 * 8
        '6': '24px',    // 3 * 8
        '8': '32px',    // 4 * 8
        '10': '40px',   // 5 * 8
        '12': '48px',   // 6 * 8
        '16': '64px',   // 8 * 8
        '20': '80px',   // 10 * 8
        '24': '96px',   // 12 * 8
        '32': '128px',  // 16 * 8
        '40': '160px',  // 20 * 8
        '48': '192px',  // 24 * 8
        '56': '224px',  // 28 * 8
        '64': '256px',  // 32 * 8
      },
      
      // Border Radius System
      borderRadius: {
        'sm': '4px',    // Small elements
        'DEFAULT': '6px', // Default for buttons, inputs
        'md': '6px',    // Medium elements  
        'lg': '8px',    // Large containers, cards
        'xl': '12px',   // Modals, major containers
        '2xl': '16px',  // Hero elements
      },
      
      // Box Shadow System for depth and focus
      boxShadow: {
        'glow': '0 0 20px rgba(0, 167, 225, 0.3)',
        'glow-lg': '0 0 40px rgba(0, 167, 225, 0.4)',  
        'neuron': '0 1px 3px rgba(0, 0, 0, 0.3), 0 1px 2px rgba(0, 0, 0, 0.24)',
        'neuron-lg': '0 10px 25px rgba(0, 0, 0, 0.3), 0 4px 6px rgba(0, 0, 0, 0.24)',
        'focus': '0 0 0 2px rgba(0, 167, 225, 0.6)',
        'neuron-focus': '0 0 0 2px rgba(0, 167, 225, 0.6)',
        'neuron-error-focus': '0 0 0 2px rgba(239, 68, 68, 0.6)',
        'neuron-focus-visible': '0 0 0 3px rgba(0, 167, 225, 0.8)',
      },
      
      // Animation System
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fade-in 0.3s ease-out',
        'slide-up': 'slide-up 0.3s ease-out',
        'scale-in': 'scale-in 0.2s ease-out',
        'synapse-pulse': 'synapse-pulse 1.5s ease-in-out infinite',
      },
      
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { 
            opacity: '1',
            boxShadow: '0 0 20px rgba(0, 167, 225, 0.3)'
          },
          '50%': { 
            opacity: '0.8',
            boxShadow: '0 0 40px rgba(0, 167, 225, 0.6)'
          },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'scale-in': {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        'synapse-pulse': {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.8' },
          '50%': { transform: 'scale(1.05)', opacity: '1' },
        },
      },
      
      // Transition timing
      transitionDuration: {
        '250': '250ms',
        '300': '300ms', 
      },
      
      transitionTimingFunction: {
        'neuron': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      
      // Maximum content width for optimal readability
      maxWidth: {
        'content': '1280px',
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    require("@tailwindcss/typography"),
    require("@tailwindcss/aspect-ratio"),
    // Custom component classes plugin
    function({ addComponents }) {
      addComponents({
        // Content width utility
        '.content-width': {
          '@apply max-w-content mx-auto': {},
        },
        
        // Button variants
        '.btn-neuron': {
          '@apply bg-neuron-glow hover:bg-neuron-glow-hover text-white font-medium rounded-md transition-colors duration-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-neuron-glow focus:ring-offset-2 focus:ring-offset-neuron-bg-primary': {},
        },
        
        '.btn-ghost': {
          '@apply bg-transparent hover:bg-neuron-bg-content text-neuron-text-secondary hover:text-neuron-text-primary font-medium rounded-md transition-colors duration-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-neuron-glow focus:ring-offset-2 focus:ring-offset-neuron-bg-primary': {},
        },
        
        '.btn-icon': {
          '@apply bg-transparent hover:bg-neuron-bg-content text-neuron-text-secondary hover:text-neuron-text-primary rounded-md transition-colors duration-300 p-2 focus:outline-none focus:ring-2 focus:ring-neuron-glow focus:ring-offset-2 focus:ring-offset-neuron-bg-primary': {},
        },
        
        // Input variants
        '.input-neuron': {
          '@apply w-full bg-neuron-bg-primary border border-neuron-border rounded-md px-3 py-2 text-neuron-text-primary placeholder-neuron-text-secondary focus:outline-none focus:ring-2 focus:ring-neuron-glow focus:border-neuron-glow': {},
        },
        
        // Shadow utilities
        '.shadow-neuron-focus': {
          'box-shadow': '0 0 0 2px rgba(0, 167, 225, 0.6)',
        },
        
        '.shadow-neuron-error-focus': {
          'box-shadow': '0 0 0 2px rgba(239, 68, 68, 0.6)',
        },
        
        '.shadow-neuron-focus-visible': {
          'box-shadow': '0 0 0 3px rgba(0, 167, 225, 0.8)',
        },
      })
    }
  ],
};
