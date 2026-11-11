// Minimal highlighting for the short Python and shell snippets on the download page: comments,
// strings and a few keywords. The input is escaped first, so the result is safe for {@html}.

const KEYWORDS = /\b(import|from|as|def|for|in|return|with|None|True|False|print)\b/g;

const escape = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function highlight(code: string): string {
  let out = "";
  let last = 0;
  // Strings first where they start earlier, so a "#" inside a string is not a comment.
  for (const match of code.matchAll(/("[^"\n]*"|'[^'\n]*'|#.*$)/gm)) {
    out += escape(code.slice(last, match.index)).replace(KEYWORDS, '<span class="k">$1</span>');
    out += `<span class="${match[0].startsWith("#") ? "c" : "s"}">${escape(match[0])}</span>`;
    last = match.index + match[0].length;
  }
  return out + escape(code.slice(last)).replace(KEYWORDS, '<span class="k">$1</span>');
}
