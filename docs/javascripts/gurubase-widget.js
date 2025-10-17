document.addEventListener("DOMContentLoaded", () => {
  // Load the GuruBase widget
  const guruScript = document.createElement("script");
  guruScript.src = "https://widget.gurubase.io/widget.latest.min.js";
  guruScript.defer = true;
  guruScript.id = "guru-widget-id";

  // Configure widget settings
  const widgetSettings = {
    "data-widget-id": "0u_XaRkpxn4Kgnc7xhIEzuWRRSvGsu5wx2sXBesL9Yw",
    "data-text": "Ask AI",
    "data-margins": JSON.stringify({ bottom: "30px", right: "30px" }),
    "data-light-mode": "auto",
    "data-name": "FlixOpt",
    "data-icon-url": "https://raw.githubusercontent.com/FlixOpt/flixopt/main/docs/images/flixopt-icon.svg",
    "data-bg-color": "#2e8a6b",
    "data-overlap-content": "true",
    "data-tooltip":
      "Ask questions about FlixOpt. Please note that questions and answers are visible anonymously to the FlixOpt team and GuruBase.",
    "data-tooltip-side": "left",

  };

  // Add widget settings as data attributes
  Object.entries(widgetSettings).forEach(([key, value]) => {
    guruScript.setAttribute(key, value);
  });

  document.body.appendChild(guruScript);
});
