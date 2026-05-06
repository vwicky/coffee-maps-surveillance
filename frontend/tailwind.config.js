/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cocoa: "#411e03",
        mocha: "#846851",
        beige: "#d6b38d",
        olive: "#8a895f",
        linen: "#d7cdc1",
      },
    },
  },
  plugins: [],
}