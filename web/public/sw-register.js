// Registers the service worker (cross-origin isolation + offline cache). On the very first visit the
// page is not yet isolated, so it reloads once as soon as the worker controls it. Skipped on the Vite
// dev server, which sends the isolation headers itself and should never serve cached code.
(function () {
  if (!("serviceWorker" in navigator) || location.port === "5173") return;
  var scope = document.currentScript.src.replace(/sw-register\.js.*$/, "");
  navigator.serviceWorker.register(scope + "sw.js", { scope: scope }).then(function (registration) {
    if (!window.crossOriginIsolated && !navigator.serviceWorker.controller) {
      navigator.serviceWorker.addEventListener("controllerchange", function () {
        if (!sessionStorage.getItem("oryx-sw-reloaded")) {
          sessionStorage.setItem("oryx-sw-reloaded", "1");
          location.reload();
        }
      });
    }
    return registration;
  });
  // A controlled page that is still not isolated (e.g. a hard reload bypassed the worker): reload
  // once through the worker.
  if (navigator.serviceWorker.controller && !window.crossOriginIsolated && !sessionStorage.getItem("oryx-sw-reloaded")) {
    sessionStorage.setItem("oryx-sw-reloaded", "1");
    location.reload();
  }
})();
