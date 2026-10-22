# Deploying epzero.dev

The site runs on Cloudflare Workers. Every push to `main` that touches the site builds it, runs
the end-to-end tests, and deploys ([`.github/workflows/site.yml`](../.github/workflows/site.yml)):

| What | Where | Why |
|---|---|---|
| Pages, scripts, styles, fonts, vectors, WebAssembly | Workers static assets (`dist/`) | Served from Cloudflare's edge without running any code; free and unmetered |
| Response headers | [`public/_headers`](public/_headers) | Cross-origin isolation (WebAssembly threads), CSP `frame-ancestors`, HSTS, `nosniff`, one-year caching for hashed bundles |
| The two CLIP models (64 and 89 MB) | R2 bucket `epzero-assets`, served by [`worker/index.js`](worker/index.js) at `/models/*` | Over the 25 MiB static-asset limit; served from the same origin so the page stays isolated without CORS, with range requests, ETags and the edge cache |
| `www.epzero.dev` | [`worker/www.js`](worker/www.js) | A 301 to the same path on `epzero.dev` |
| The old GitHub Pages address | the `redirect` job | Forwards `lazeeez.github.io/<repo>/<path>` to `epzero.dev/<path>` |

Unknown paths get the app's not-found page with a real 404 status; `/try/` redirects to `/try`.

## One-time setup

1. **API token.** In the Cloudflare dashboard: *My Profile → API Tokens → Create Token*, start from
   the *Edit Cloudflare Workers* template, and make sure it has:
   - Account › Workers Scripts › Edit
   - Account › Workers R2 Storage › Edit
   - Account › Account Settings › Read
   - Zone (`epzero.dev`) › Workers Routes › Edit
   - Zone (`epzero.dev`) › DNS › Edit (the custom domains create their own DNS records)
   - Zone (`epzero.dev`) › Zone › Read
2. **Account ID.** On the dashboard's overview page for `epzero.dev`, in the right-hand column.
3. **GitHub secrets.** In the repository: *Settings → Secrets and variables → Actions → New
   repository secret*: `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.
4. **Deploy.** *Actions → site → Run workflow* (or push). The first run creates the R2 bucket,
   deploys both Workers, attaches `epzero.dev` and `www.epzero.dev` (DNS records included),
   uploads the models, and smoke-tests the live site. Later runs upload only what changed.

Without the secrets the Cloudflare job skips itself with a warning; everything else still runs.

Recommended zone settings (*SSL/TLS* and *Speed* in the dashboard): *Always Use HTTPS* on,
*Minimum TLS* 1.2, *Brotli* on (the default). The site sends its own HSTS header, and `.dev`
domains are HTTPS-only in browsers anyway.

## Running it locally the way Cloudflare does

```bash
npm run build
npx wrangler dev                      # http://localhost:8787 with the production routing and headers
# the models, into a local R2 bucket (once):
cd dist && for f in $(find models -type f); do npx wrangler r2 object put "epzero-assets/$f" --file "$f" --local -c ../wrangler.jsonc; done
E2E_BASE_URL=http://localhost:8787/ npx playwright test --project=chromium
```

`npx wrangler deploy --dry-run` checks the configuration without deploying.

## Search engines

What the site already does: every page's HTML has its own title, description, canonical address,
link-preview tags and a no-JavaScript fallback with its heading and links; there is structured
data (`WebSite`, `WebApplication`, `Person`), a sitemap written at each build with the build date,
`robots.txt` pointing at it, and a `noindex` 404 page ([`e2e/seo.spec.ts`](e2e/seo.spec.ts) tests
all of it). To get it indexed:

1. **Google Search Console** (search.google.com/search-console): *Add property → Domain →
   `epzero.dev`*. For a domain on Cloudflare, Google offers to add its verification TXT record
   for you; otherwise add the `google-site-verification=...` TXT record under *DNS* in Cloudflare
   (name `@`), then *Verify*.
2. *Sitemaps*: submit `https://epzero.dev/sitemap.xml`.
3. *URL inspection*: inspect `https://epzero.dev/`, `/try`, `/benchmarks`, `/cluster` and `/play`,
   and *Request indexing* for each. Indexing usually takes a few days.
4. **Bing Webmaster Tools** (bing.com/webmasters): *Import from Google Search Console*. That covers
   Bing, DuckDuckGo and Yahoo too.
5. **Cloudflare → Caching → Configuration → Crawler Hints**: on. Cloudflare then tells search
   engines (IndexNow) when pages change, so updates are picked up without waiting for a crawl.
6. Check the structured data with Google's Rich Results Test (search.google.com/test/rich-results).
7. Links from elsewhere are what search engines weigh most for a new domain: put `epzero.dev` in
   the GitHub repository's *About → Website* field, on LinkedIn and your résumé, and in posts that
   share the project.
