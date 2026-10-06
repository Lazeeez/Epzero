<script lang="ts">
  let { onclose }: { onclose: () => void } = $props();
  let dialog: HTMLDialogElement | undefined = $state();

  $effect(() => {
    if (!dialog) return;
    dialog.showModal();
    return () => dialog?.close();
  });

  const groups: { title: string; keys: [string[], string][] }[] = [
    {
      title: "Search",
      keys: [
        [["/"], "Focus the search box"],
        [["Enter"], "Search"],
      ],
    },
    {
      title: "Debugger",
      keys: [
        [["Space"], "Play / pause the replay"],
        [["←", "→"], "Step back / forward"],
        [["Shift", "←→"], "Step 10 at a time"],
      ],
    },
    {
      title: "View",
      keys: [
        [["+", "−"], "Zoom in / out"],
        [["0"], "Reset the view"],
        [["R"], "Auto-rotate on / off"],
        [["L"], "Isolate the next layer (Shift: previous)"],
        [["H"], "Free look: hide or show the panels"],
        [["?"], "This list"],
      ],
    },
  ];
</script>

<dialog class="shortcuts card" aria-labelledby="shortcuts-title" bind:this={dialog} {onclose}>
  <header>
    <h2 id="shortcuts-title">Keyboard shortcuts</h2>
    <button class="close" onclick={onclose} aria-label="Close">×</button>
  </header>
  <div class="groups">
    {#each groups as group (group.title)}
      <section>
        <h3>{group.title}</h3>
        <dl>
          {#each group.keys as [keys, action] (action)}
            <dt>{#each keys as key (key)}<kbd>{key}</kbd>{/each}</dt>
            <dd>{action}</dd>
          {/each}
        </dl>
      </section>
    {/each}
  </div>
  <p class="mouse">Mouse: drag to orbit · right-drag to pan · scroll to zoom toward the cursor · double-click to fly in.</p>
</dialog>

<style>
  .shortcuts {
    width: min(620px, 100% - 32px);
    max-height: calc(100dvh - 32px);
    margin: auto;
    padding: 22px;
    overflow: auto;
    background: var(--surface-solid);
    color: var(--text);
  }
  .shortcuts::backdrop {
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(4px);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
  }
  h2 {
    font-size: 19px;
  }
  .close {
    width: 32px;
    height: 32px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--surface);
    color: var(--text-2);
    font-size: 18px;
    cursor: pointer;
  }
  .groups {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 18px;
  }
  h3 {
    margin-bottom: 8px;
    color: var(--accent-strong);
    font: 600 11.5px var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  dl {
    display: grid;
    gap: 8px;
    margin: 0;
  }
  dt {
    display: flex;
    gap: 4px;
  }
  dd {
    margin: -4px 0 4px;
    color: var(--text-2);
    font-size: 13px;
  }
  .mouse {
    margin: 18px 0 0;
    color: var(--muted);
    font-size: 12.5px;
  }
</style>
