// Light/dark toggle. Runs from dashboard/assets/ (Dash serves it automatically).
// data-theme on <html> overrides the OS preference; absent means "follow the OS".
(function () {
  var KEY = "theme", root = document.documentElement;
  try {
    var saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch (e) { /* storage blocked: the choice just won't persist */ }

  // Delegated, because Dash renders the button after this script loads.
  document.addEventListener("click", function (e) {
    if (!e.target.closest("#theme-toggle")) return;
    var dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    var next = dark ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem(KEY, next); } catch (e) {}
  });
})();
