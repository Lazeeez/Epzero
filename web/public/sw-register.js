// Registers the service worker, which keeps the site offline-capable and, on a host that can't send
// the COOP/COEP headers (production on Cloudflare does), gives it cross-origin isolation, which
// WebAssembly threads need. Without the headers a page is only isolated when the worker served it, so:
//   - first visit: the worker isn't in control yet; reload once it takes control;
//   - hard refresh (Ctrl/Cmd+Shift+R): the browser bypassed the worker; a normal reload goes back
//     through it.
// A browser that can't be isolated never loops: after 2 reloads in a row that didn't produce an
// isolated page, it stops and the site runs single-threaded and says so. The count resets whenever a
// page does load isolated, so a later hard refresh can always recover. State for the app is on
// window.epzeroIsolation ("isolated" | "reloading" | "waiting" | "unavailable"), with an
// "epzero-isolation" event when it settles. Skipped on the Vite dev server, which sends the headers.
(function () {
  var state = function (value) {
    window.epzeroIsolation = value;
    window.dispatchEvent(new Event("epzero-isolation"));
  };
  if (window.crossOriginIsolated) {
    window.epzeroIsolation = "isolated";
    try {
      sessionStorage.removeItem("epzero-sw-reloads"); // reloading worked: reset the loop guard
    } catch (e) {
      /* storage blocked */
    }
  } else if (!("serviceWorker" in navigator) || !window.isSecureContext || window.crossOriginIsolated === undefined) {
    window.epzeroIsolation = "unavailable"; // no service workers, or no isolation support at all
  } else {
    window.epzeroIsolation = "waiting";
  }
  if (!("serviceWorker" in navigator) || location.port === "5173") return;

  function reload() {
    try {
      // Reloads since the last isolated page load: each one that didn't help is a step toward a loop.
      var attempts = JSON.parse(sessionStorage.getItem("epzero-sw-reloads") || "[]");
      if (attempts.length >= 2) {
        state("unavailable"); // reloading isn't working here: stay single-threaded rather than loop
        return;
      }
      attempts.push(Date.now());
      sessionStorage.setItem("epzero-sw-reloads", JSON.stringify(attempts));
    } catch (e) {
      state("unavailable"); // no sessionStorage: can't guard against loops, so don't reload
      return;
    }
    window.epzeroIsolation = "reloading";
    location.reload();
  }

  var scope = document.currentScript.src.replace(/sw-register\.js.*$/, "");
  navigator.serviceWorker
    .register(scope + "sw.js", { scope: scope })
    .then(function (registration) {
      if (window.crossOriginIsolated || window.epzeroIsolation === "unavailable") return;
      if (navigator.serviceWorker.controller || registration.active) {
        reload(); // controlled but not isolated, or a hard refresh bypassed an active worker
      } else {
        navigator.serviceWorker.addEventListener("controllerchange", reload); // first visit
      }
    })
    .catch(function () {
      state("unavailable"); // e.g. private browsing: single-threaded, no offline cache
    });
})();
