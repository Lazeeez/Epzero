// The engineering article (docs/blog/vector-search-is-a-memory-problem.md), as HTML for the site's
// /blog page. Converted at build time (and by the dev server), so no Markdown library ships to
// visitors. The site version ends with the site's own pages instead of the repository, which the
// site doesn't link to. Tables get a scrolling wrapper, and those of three or more columns carry
// each cell's header as data-label, so phones can show their rows as labelled cards.
import { readFile } from "node:fs/promises";
import { Marked, Renderer, type Token } from "marked";

export interface Article {
  title: string;
  subtitle: string;
  html: string;
  words: number;
  /** Plain text, for the no-JavaScript fallback's summary. */
  text: string;
}

const SOURCE = new URL("../../docs/blog/vector-search-is-a-memory-problem.md", import.meta.url);

const CLOSING =
  "You can search 30,000 photos with the same engine, compiled to WebAssembly, [in your browser](/try), " +
  "add your own photos there, or [benchmark it on your own embeddings](/benchmarks#lab).";

export async function loadArticle(): Promise<Article> {
  let markdown = (await readFile(SOURCE, "utf8")).replace(/\r\n/g, "\n");
  const title = /^# (.+)$/m.exec(markdown)?.[1] ?? "Vector search is a memory problem";
  const subtitle = /^\*([^*\n]+)\*$/m.exec(markdown)?.[1] ?? "";
  markdown = markdown.replace(/^# .+\n+/, "").replace(/^\*[^*\n]+\*\n+/, "");
  // The last paragraph points at the repository and its files: the site says where to go instead.
  markdown = markdown.replace(/\nThe code, the benchmark scripts[\s\S]*$/, `\n${CLOSING}\n`);
  // Links to this site become site-relative; any other link to the repository becomes plain text.
  markdown = markdown
    .replace(/\]\(https:\/\/epzero\.dev\/?([^)]*)\)/g, (_, path: string) => `](/${path})`)
    .replace(/\[([^\]]+)\]\((?:https:\/\/github\.com[^)]*|\.\.?\/[^)]*)\)/g, "$1");
  const marked = new Marked({ gfm: true });
  const base = new Renderer();
  marked.use({
    renderer: {
      table(token) {
        if (token.header.length < 3) return `<div class="table-wrap">${base.table.call(this, token)}</div>`;
        const inline = (cell: { tokens: Token[] }) => this.parser.parseInline(cell.tokens);
        const labels = token.header.map((cell) => inline(cell).replace(/<[^>]+>/g, "").replace(/"/g, "&quot;"));
        const head = token.header.map((cell) => `<th>${inline(cell)}</th>`).join("");
        const rows = token.rows
          .map((row) => `<tr>${row.map((cell, i) => `<td data-label="${labels[i]}">${inline(cell)}</td>`).join("")}</tr>`)
          .join("\n");
        return `<div class="table-wrap stack"><table>\n<thead><tr>${head}</tr></thead>\n<tbody>${rows}</tbody>\n</table></div>\n`;
      },
      heading({ tokens, depth }) {
        const text = this.parser.parseInline(tokens);
        const id = text
          .replace(/<[^>]+>/g, "")
          .toLowerCase()
          .replace(/&[a-z]+;/g, "")
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-|-$/g, "");
        return `<h${depth} id="${id}">${text}</h${depth}>\n`;
      },
    },
  });
  const html = await marked.parse(markdown);
  const text = html
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-z#0-9]+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return { title, subtitle, html, words: text.split(" ").length, text };
}
