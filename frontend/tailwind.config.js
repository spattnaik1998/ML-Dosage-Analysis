/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary hospital blue (inspired by Boston Medical Center)
        'hospital': {
          50: '#f0f7ff',
          100: '#e0efff',
          200: '#b9ddff',
          300: '#7cc3ff',
          400: '#3aa6ff',
          500: '#0d8ae8',
          600: '#006dc6',
          700: '#00559d',
          800: '#004680',
          900: '#003a6a',
          950: '#002647',
        },
        // Medical green accent
        'medical': {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        // Warm accent for critical alerts
        'care': {
          50: '#fef3f2',
          100: '#fee4e2',
          200: '#fecdd6',
          300: '#fda4af',
          400: '#fb7185',
          500: '#f43f5e',
          600: '#e11d48',
          700: '#be123c',
          800: '#9f1239',
          900: '#881337',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Poppins', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'medical': '0 2px 8px 0 rgba(0, 109, 198, 0.08)',
        'medical-lg': '0 8px 24px 0 rgba(0, 109, 198, 0.12)',
      }
    },
  },
  plugins: [],
}
