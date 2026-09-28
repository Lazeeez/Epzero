import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles/global.css";
import { mount } from "svelte";
import App from "./App.svelte";
import { router } from "./lib/router.svelte";
import { restoreTheme } from "./lib/theme";

// After a deploy, a tab still running the previous build can ask for a lazily loaded chunk that no
// longer exists. Reload once to pick up the new build instead of breaking.
addEventListener("vite:preloadError", (event) => {
  try {
    if (sessionStorage.getItem("oryx-chunk-reload")) return;
    sessionStorage.setItem("oryx-chunk-reload", "1");
  } catch {
    return;
  }
  event.preventDefault();
  location.reload();
});

restoreTheme();
router.start();
mount(App, { target: document.getElementById("app")! });
