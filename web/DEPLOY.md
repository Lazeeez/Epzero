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
