// The engineering article, converted from Markdown at build time (vite.config.ts, scripts/article.ts).
declare module "virtual:article" {
  const article: { title: string; subtitle: string; html: string; words: number };
  export default article;
}
