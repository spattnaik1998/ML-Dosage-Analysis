/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary red (vibrant medical red)
        'hospital': {
          50: '#ffebee',
          100: '#ffcdd2',
          200: '#ef9a9a',
          300: '#e57373',
          400: '#ef5350',
          500: '#f44336',
          600: '#dc143c',
          700: '#c62828',
          800: '#b71c1c',
          900: '#8b0000',
          950: '#6d0000',
        },
        // Black/dark accent
        'medical': {
          50: '#f5f5f5',
          100: '#e0e0e0',
          200: '#bdbdbd',
          300: '#757575',
          400: '#424242',
          500: '#303030',
          600: '#1f1f1f',
          700: '#1a1a1a',
          800: '#0f0f0f',
          900: '#0a0a0a',
        },
        // Red accent for critical alerts
        'care': {
          50: '#ffebee',
          100: '#ffcdd2',
          200: '#ef9a9a',
          300: '#e57373',
          400: '#ef5350',
          500: '#f44336',
          600: '#e53935',
          700: '#d32f2f',
          800: '#c62828',
          900: '#b71c1c',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Poppins', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'medical': '0 2px 8px 0 rgba(220, 20, 60, 0.12)',
        'medical-lg': '0 8px 24px 0 rgba(220, 20, 60, 0.18)',
      }
    },
  },
  plugins: [],
}
