/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cream: "#F7F2EC",
        "dark-cream": "#F0E8DF",
        taupe: "#EAE0D4",
        espresso: "#2C1A0E",
        "medium-brown": "#8A5E4A",
        terracotta: "#C4714A",
        tan: "#C9B4A3",
        "medium-tan": "#DDD0BE",
        error: "#A63D2F",
        "error-bg": "#F5EAE8",
        success: "#5C7A5A",
        "success-bg": "#E8EDE5"
      },
      fontFamily: {
        display: ["Playfair Display", "serif"],
        body: ["Jost", "sans-serif"],
      }
    },
  },
  plugins: [],
}

