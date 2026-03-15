/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        neon: {
          cyan: 'rgb(0, 212, 255)',
          soft: 'rgb(157, 196, 255)',
        },
      },
      fontFamily: {
        sans: ['"Inter"', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      animation: {
        marquee: 'marquee 20s linear infinite',
        'spin-slow': 'spin 8s linear infinite',
        fadeIn: 'fadeIn 0.5s ease forwards',
        fadeInUp: 'fadeInUp 0.5s ease forwards',
        fadeInDown: 'fadeInDown 0.5s ease forwards',
        slideInLeft: 'slideInLeft 0.5s ease forwards',
        slideInRight: 'slideInRight 0.5s ease forwards',
        float: 'float 3s ease-in-out infinite',
        'bounce-subtle': 'bounce 2s ease-in-out infinite',
        shimmer: 'shimmer 1.5s infinite',
        countUp: 'countUp 0.5s ease forwards',
        confetti: 'confetti 1s ease-out forwards',
      },
      keyframes: {
        marquee: { '0%': { transform: 'translate(0)' }, to: { transform: 'translate(-50%)' } },
        fadeIn: { '0%': { opacity: '0' }, to: { opacity: '1' } },
        fadeInUp: { '0%': { opacity: '0', transform: 'translateY(20px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        fadeInDown: { '0%': { opacity: '0', transform: 'translateY(-20px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        slideInLeft: { '0%': { opacity: '0', transform: 'translateX(-30px)' }, to: { opacity: '1', transform: 'translateX(0)' } },
        slideInRight: { '0%': { opacity: '0', transform: 'translateX(30px)' }, to: { opacity: '1', transform: 'translateX(0)' } },
        float: { '0%, to': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-10px)' } },
        shimmer: { '0%': { 'background-position': '-200% 0' }, to: { 'background-position': '200% 0' } },
        countUp: { '0%': { opacity: '0', transform: 'translateY(10px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        confetti: { '0%': { transform: 'translateY(0) rotate(0)', opacity: '1' }, to: { transform: 'translateY(-100vh) rotate(720deg)', opacity: '0' } },
      },
    },
  },
  plugins: [],
}
