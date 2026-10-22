// Free look on the playground: every panel and control is hidden, leaving the 3D graph and the top
// bar. Shared state, because the page hides its panels, the view controls' reset button turns it on,
// and the top bar holds the way back.
import { tick } from "svelte";

export const freeLook = $state({ on: false });

/** The top bar's "Show panels" button: the way back, so it takes the keyboard focus in free look. */
export const FREE_LOOK_TOGGLE = "free-look-toggle";
/** The view controls' reset button, which resets the camera and turns free look on. */
export const VIEW_RESET = "view-reset";

/** Turns free look on or off. The keyboard focus follows the control that undoes it. */
export async function setFreeLook(on: boolean) {
  if (on === freeLook.on) return;
  const focused = document.activeElement?.id;
  freeLook.on = on;
  await tick();
  if (focused === VIEW_RESET || focused === FREE_LOOK_TOGGLE) {
    document.getElementById(on ? FREE_LOOK_TOGGLE : VIEW_RESET)?.focus();
  }
}
