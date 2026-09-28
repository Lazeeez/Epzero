import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles/global.css";
import { mount } from "svelte";
import App from "./App.svelte";
import { router } from "./lib/router.svelte";
import { restoreTheme } from "./lib/theme";

restoreTheme();
router.start();
mount(App, { target: document.getElementById("app")! });
