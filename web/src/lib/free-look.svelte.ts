// Free look on the playground: every panel and control is hidden, leaving the 3D graph and the top
// bar. Shared state, because the page hides its panels and the top bar holds the way back.
export const freeLook = $state({ on: false });

/** The top bar's toggle, which takes the keyboard focus when the panels (and their button) go. */
export const FREE_LOOK_TOGGLE = "free-look-toggle";
