# Unusual Activity Watchdog

**Synapse 1.0 (SIT Flagship Hackathon) — AI in FinTech track**

> Build a system that continuously watches live buying/selling activity for a
> stock and raises an alert when something breaks the normal pattern — a
> sudden burst of trades, a price jump with no news behind it, several stocks
> moving together in a suspicious way — so a human analyst can investigate.

This repo is a working prototype of exactly that, built up into a full
surveillance platform: a **dual-AI ensemble** (Isolation Forest + neural
autoencoder) for per-stock anomalies, a **cross-stock correlation-break
engine** (with a concentration check that tells a genuine small cluster
apart from an ordinary market-wide move), exchange-grade **order-flow
manipulation surveillance** (spoofing / layering / quote stuffing / wash
trading), a **live animated replay mode** with a composite market-stress
gauge, a **3D anomaly landscape**, a **correlation network diagram**,
browser **voice alerts**, and one-click **PDF incident reports** — all
wrapped in a Streamlit dashboard with a built-in simulated market (so the
demo works reliably regardless of market hours, network access, or luck)
plus a real yfinance live-data path with a genuine background **autorefresh**
(Live mode silently re-pulls and re-scores on a timer, so "continuously
watches" is a true statement with a fixed, quotable worst-case detection
latency, not just "whenever someone last clicked refresh").

**v2.1 note:** this codebase went through an independent technical review
(scored 64/100, "partially ready") before this pass, which found five
concrete, well-scoped gaps: no autorefresh, a presence-only (not
relevance-scored) news check, no numeric per-alert confidence score, a
correlation engine that couldn't yet tell a genuine coordinated cluster
from a broad market-wide move, and a couple of repo-hygiene items. All five
are addressed in this version — see the "What changed in the review pass"
callouts throughout this README for exactly what and why.

**v2.2 note (architecture upgrade):** a follow-up independent audit
proposed a 4-sprint upgrade plan, implemented in full without a
frontend/API rewrite -- still the same Python/Streamlit codebase:

- **Two new independent detection signals**: `detection/regime.py` (a
  price-level "breadth" check -- what fraction of the watchlist moved
  unusually on the same bar -- independent of and complementary to
  `correlation_watch.py`'s correlation-level concentration check) and
  `detection/momentum.py` (a short-vs-long EWMA crossover, itself a
  z-score outlier against its own history -- a trend-divergence signal the
  Isolation Forest/autoencoder ensemble doesn't look for, surfaced as a new
  `momentum_shift` alert kind).
- **One composite 0-100 anomaly score** (`detection/score.py`) -- a
  documented, hand-set weighted blend of six signal components -- replaces
  ad hoc per-kind confidence math for every per-ticker alert, alongside a
  *second*, deliberately separate number, `agreement_pct` (how many
  independent detectors concur), so "how unusual" and "how many methods
  agree" are never collapsed into one figure.
- **Native `st.fragment(run_every=...)` autorefresh** replaces the
  third-party `streamlit-autorefresh` package: the Command Center header
  and Alert Feed silently re-pull and re-score on a configurable 10/20/60s
  interval *without* rerunning the whole page (heavier tabs like the 3D
  landscape and order-flow simulation no longer recompute every tick).
  Live fetches retry with backoff before falling back to simulated, and a
  stale-data check degrades the "continuous monitoring" banner to a
  "data delayed" warning once the last successful refresh is too old.
- **SQLite-backed alert persistence + an investigation workflow**
  (`data/store.py`, `services/alert_service.py`): every alert gets a
  stable identity across pipeline reruns and a status --
  `NEW -> INVESTIGATING -> RESOLVED` / `FALSE POSITIVE` -- settable from
  the Alert Feed or the new Alert History page. **Honest caveat:** on a
  host with no persistent disk (e.g. Render's free tier), this survives
  the running instance only -- it's wiped on redeploy or cold restart. It
  is still a real upgrade over "nothing persists, ever."
- **Three new pages** (Streamlit's multipage `pages/` convention, shared
  pipeline logic factored into `services/pipeline_service.py` so a page
  never has to re-run `app.py`'s own script to get data): a **Stock
  Investigation Terminal** (per-ticker price/volume/momentum/news/
  correlation/alert-history deep dive), **Alert History & Analytics**
  (trends, breakdowns by kind/severity/ticker, investigation-outcome
  rate), and **System Health** (a fixed battery of OK/WARN/FAIL checks
  against a live pipeline run, so a demo presenter -- or an operator --
  can tell "quiet market" apart from "silently broken detector").
- **3-state news classification** (`services/news_service.py`): every
  price-jump alert's news check is now exactly one of `NEWS EXPLAINS
  MOVEMENT`, `POSSIBLE NEWS-MOVEMENT MISMATCH` (a headline exists nearby
  but didn't clear the relevance bar), or `NO RELEVANT NEWS DETECTED`,
  with a human-readable headline-to-move time gap.
- **Command Center reassembly**: a real-time market heatmap (latest-bar
  return/volume z-scores across the whole watchlist, refreshing with the
  header), a **click-through correlation network** (click a node for an
  inline ticker snapshot and a jump straight into the Stock Terminal), and
  a **demo-mode banner** with a "trigger synthetic anomaly now" button
  (Simulated mode only) that stamps an unmistakable move onto the current
  scenario's latest bar so a presenter can make something happen on
  demand.
- **Consolidated design tokens** (`utils/design_tokens.py`,
  `utils/formatting.py`): one source of truth for the dark-mode palette,
  Plotly theme, badges, and empty/error/loading copy, shared by app.py and
  every page under `pages/` instead of each redefining its own.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Defaults to **Simulated demo** mode, which
always shows a full set of detected anomalies — good for a live pitch where
you can't wait for something unusual to actually happen on the real market.
Switch to **Live / historical (yfinance)** in the sidebar to pull real
intraday data (needs normal internet access — see *Known limitations* below).
In Live mode, **Auto-refresh (continuous monitoring)** is on by default: a
native `st.fragment(run_every=...)` silently re-pulls fresh bars and
re-scores the Command Center header and Alert Feed on a sidebar-configurable
10/20/60s interval (`config.py: live_autorefresh_seconds` sets the default,
20s) with no click needed and no full-page rerun, and the header shows the
resulting worst-case detection latency (bar interval + refresh interval) as
a fixed, quotable number -- degrading to a "data delayed" warning if the
last successful refresh gets stale.

Run the sanity-check test suite (checks every detector against known,
injected ground truth):

```bash
python3 tests/test_pipeline.py
```

## What's in the dashboard

`app.py` is the **Command Center** (7 tabs below); three more pages live
under `pages/` (Streamlit's multipage convention -- they appear in the
sidebar automatically).

| Tab (Command Center) | What it shows |
|---|---|
| **Alert Feed** | Ranked, deduplicated alerts with a numeric 0-100 confidence score, `agreement_pct`, AI-agreement badges, SHAP driver charts, 3-state news classification, per-alert investigation-status controls, and a one-click PDF incident report per alert |
| **Price & Volume** | Full per-ticker detail view: price, volume, return z-score with alert thresholds drawn in |
| **Correlation Monitor** | Rolling correlation vs. baseline over time, plus current-vs-baseline correlation heatmaps -- flagged breaks are further split into a genuine `correlated_group_move` cluster vs. a broad `market_wide_move`, shown per-alert as a concentration ratio |
| **3D & Network** | A **click-through** live correlation network diagram (click a node for an inline ticker snapshot + a jump to the Stock Terminal; flagged clusters highlighted) and a 3D scatter of every bar in anomaly-feature space |
| **Live Replay** | Scrubs/auto-plays through the scenario bar-by-bar, revealing alerts and a market-stress gauge only up to the current point in time -- watch detection happen instead of reading a static table |
| **Order-Flow Surveillance** | Simulated order-event stream + rule-based detectors for spoofing, layering, quote stuffing, and wash trading |
| **Detection Accuracy** | Live recall check against the simulated scenario's known, injected ground truth |

The Command Center header also carries a real-time market heatmap and (in
Simulated mode) a demo banner with a "trigger synthetic anomaly now" button.

| Page (`pages/`) | What it shows |
|---|---|
| **Stock Investigation Terminal** | One ticker's full picture: price/volume/return-zscore detail, the momentum (EWMA crossover) signal, a news-relevance timeline, correlation-cluster context, and its own alert/investigation history |
| **Alert History & Analytics** | Filterable log of every persisted alert, trends over time, breakdowns by kind/severity/ticker, and the resolved-vs-false-positive investigation-outcome rate |
| **System Health** | A fixed battery of OK/WARN/FAIL checks (data source, autorefresh config, detection ensemble, correlation engine, news scorer, SQLite persistence, alert-volume sanity) against a live pipeline run |

## Why so many separate signals, not one model

The problem statement names three distinct bar-level patterns, and a real
surveillance desk's job goes further than that -- and each layer genuinely
needs its own detection strategy, not a bigger version of the same model:

| Pattern | Why it needs its own approach |
|---|---|
| **Sudden burst of trades** | A per-stock question: is *this* volume unusual *for this stock*. A global model would just learn "big-cap = normal." |
| **Price jump with no news** | Two-part: is the move statistically unusual, *and* is there a news catalyst. Needs a price-anomaly signal AND a news lookup, fused together. |
| **Several stocks moving together suspiciously** | Invisible to any per-stock model by definition — each stock might look perfectly ordinary alone. Needs to look across the whole watchlist at once. |
| **Spoofing / layering / wash trading** | Invisible at the bar level entirely — these are defined by order *lifecycle* (placed, then cancelled before filling; bought then sold moments later), which OHLCV bars don't even record. Needs the finer-grained order-event stream. |

## Architecture

```
data/synthetic.py          -- simulated multi-stock tape with injected, known anomalies
data/live_feed.py          -- real intraday data via yfinance + relevance-scored live news lookup
data/orderbook_synthetic.py -- simulated order-EVENT stream (spoofing/layering/stuffing/wash trades)
data/store.py                -- SQLite alert persistence + investigation-workflow CRUD (parameterized queries)
data_source.py               -- live -> simulated fallback DECISION logic, split out so it's unit-testable
        |
        v
features.py                 -- rolling z-scores (volume, return) + volatility, per ticker
        |
        v
models/anomaly_model.py     -- per-ticker Isolation Forest, SHAP-explained
models/autoencoder_model.py -- per-ticker neural autoencoder (reconstruction-error anomaly score)
models/ensemble.py          -- combines both into one score + a "dual-model consensus" flag
detection/regime.py          -- price-level breadth check (market-wide event vs. genuine small cluster)
detection/momentum.py        -- short-vs-long EWMA crossover, z-scored against its own history
detection/score.py           -- one composite 0-100 anomaly score (six weighted components) + agreement_pct
correlation_watch.py         -- cross-stock rolling-correlation break detector + cluster-vs-market-wide concentration check
news_relevance.py            -- TF-IDF + keyword relevance scoring for the news check (not just "does a headline exist")
order_flow.py                -- rule-based spoofing/layering/stuffing/wash-trade detectors
stress_index.py              -- composite 0-100 market-stress index
        |
        v
alert_engine.py              -- fuses all signals + relevance-scored news check into one ranked feed, with a 0-100 confidence score per alert
services/pipeline_service.py -- shared pipeline orchestration (resolve data -> features -> ensemble -> alerts), used by app.py AND every page/
services/alert_service.py    -- investigation-workflow service on top of data/store.py (the only module that touches sqlite3 directly)
services/news_service.py     -- 3-state news classification (EXPLAINS / MISMATCH / NONE) + headline-to-move gap text
services/health_service.py   -- OK/WARN/FAIL checks against a live pipeline run
utils/design_tokens.py       -- single source of truth for the dark-mode palette, Plotly theme, badges
utils/formatting.py          -- shared empty/error/loading copy + display-formatting helpers
report_generator.py          -- one-click PDF incident report per alert
viz_extra.py                 -- 3D anomaly landscape, click-through correlation network, voice-alert snippet
        |
        v
app.py                       -- Command Center (7 tabs, see above): st.fragment autorefresh, heatmap, demo banner
pages/1_Stock_Terminal.py    -- per-ticker deep dive
pages/2_Alert_History.py     -- historical analytics + investigation-outcome stats
pages/3_System_Health.py     -- system health panel
```

### 1. Per-ticker anomaly detection: a two-model ensemble (`models/`)

An **Isolation Forest** is trained separately for each ticker on three
rolling features (`return_zscore`, `volume_zscore`, `volatility`) — not
pooled across the watchlist, because different stocks trade at very
different scales and a pooled model would just learn "which stock is this"
rather than "is this unusual for this stock."

Isolation Forest's `contamination` parameter guarantees it flags a fixed ~5%
of bars as "relatively" unusual *even on pure noise* — that's inherent to how
the algorithm ranks points, not a bug. To avoid alert fatigue from that noise
floor, every flagged point is also gated against an **absolute** z-score
threshold (`config.py`: `volume_zscore_alert`, `price_jump_sigma_alert`)
before it becomes a real alert.

A second, **architecturally independent** model — a compact **neural
autoencoder** (`models/autoencoder_model.py`, scikit-learn `MLPRegressor`
trained to reconstruct a short sliding window of its own input features,
scored by reconstruction error) — runs in parallel. `models/ensemble.py`
combines both into one score and flags **"dual-model consensus"** when both
independently agree a bar is anomalous. That agreement is the strongest
signal the system produces: it's surfaced everywhere (a distinct badge in
the alert feed, an automatic severity upgrade to High, a diamond marker in
the 3D landscape) because two structurally different models agreeing rules
out either one's individual blind spots. The autoencoder is deliberately an
MLP rather than an LSTM/Transformer — it trains in well under a second per
ticker and adds no heavy dependency (no PyTorch/TensorFlow), which matters
for staying inside a free-tier hosting deploy, while still being a real,
independently-trained neural net rather than a second copy of the same idea.

Every flagged point is also explained with **SHAP** (`TreeExplainer` on the
Isolation Forest), so each alert says *which feature drove it*, not just
"this looks weird."

### 2. Cross-stock correlation-break detection (`correlation_watch.py`)

At each bar, a short rolling pairwise-correlation matrix (recent behavior) is
compared against a much longer trailing baseline (normal co-movement). Two
design choices that made this actually work in testing:

- **Top-k pairwise deltas, not the average across all pairs.** Averaging
  across every pair in the watchlist dilutes a genuine 3-stock cluster into
  noise (only 3 of e.g. 21 pairs actually move). Looking at the top 3
  pairwise deltas keys directly onto "a small group moving together."
- **Expanding-history z-score, not a short rolling z-score.** A short
  rolling baseline would include the anomalous bars themselves, dragging its
  own "normal" up and masking the event. An expanding (whole-history) mean/
  std barely moves for one localized event, so the anomaly still stands out.

**What changed in the review pass — concentration ratio.** The design above
correctly solves "don't dilute a genuine cluster into noise," but it left a
real gap: a broad market-wide event (a Fed announcement, an index-level
selloff) can also make the top-3 pairwise delta look unusual, since
essentially every pair's correlation rises together — including,
coincidentally, the top 3. Nothing checked whether the flagged "cluster" was
actually *concentrated* versus the rest of the book having moved together
too. A `concentration_ratio` column now compares the top-k cluster signal
against the mean |delta| across the FULL correlation matrix at that same
moment: well above `correlation_concentration_ratio_threshold` (1.8x by
default) means a small subset moved much more than everything else (a
genuine `correlated_group_move`); at or below it, the whole watchlist moved
together, and it's now labeled a lower-urgency `market_wide_move` instead —
so a broad market move and a genuine coordinated cluster no longer produce
the same alert. A dedicated regression test
(`tests/test_pipeline.py::test_market_wide_move_not_flagged_as_cluster`)
injects a uniform shock across the ENTIRE watchlist and asserts it is
classified as `market_wide_move`, never `correlated_group_move`.

Validated on the simulated generator: **~87% of injected correlated-group
events detected with the correct tickers named** (unchanged by the
concentration-ratio fix), across 15 random seeds.

### 3. Order-flow manipulation surveillance (`order_flow.py` + `data/orderbook_synthetic.py`)

Everything above works at the OHLCV bar level, which is the finest grain
free data provides. Real exchange surveillance desks watch something finer
still: the order-by-order event stream, because several classic
manipulation patterns are only visible there. No free source hands you real
per-order, per-trader data (it's proprietary exchange data), so
`data/orderbook_synthetic.py` generates a labeled synthetic one, and
`order_flow.py` runs the same rule shapes real surveillance systems use as
first-line detectors:

- **Spoofing** — a large order (≥8x the trader's own median size) placed and
  cancelled within 45s without ever filling.
- **Layering** — the same trader stacking 3+ orders across price levels on
  one side within 15s, all pulled together with zero fills.
- **Quote stuffing** — 25+ order messages from one trader within a 5-second
  window — far beyond any plausible manual trading rate.
- **Wash trading** — the same trader buying and selling the same size at the
  same price within 10 seconds, netting no real position change.

All four injected patterns are caught in testing (`tests/test_pipeline.py::test_order_flow_surveillance`).

### 4. News check, alert fusion, and the stress index

Uses yfinance's built-in `Ticker.news` (no separate API/key needed) to check
for a headline near a flagged price move, separating "explained" from
"unexplained" jumps (a synthetic news table plays the same role in demo
mode). `alert_engine.py` fuses every signal into one ranked feed and merges
consecutive alerts of the same kind/ticker(s) into a single event, so a
sustained anomaly reads as one event, not N near-duplicates.

**What changed in the review pass — relevance, not presence.** The news
check used to ask only "does any headline exist for this ticker nearby,"
which treats an unrelated wire story the same as a genuine market-moving
one — the weakest link against the problem statement's own hardest clause
("a price jump with no news behind it"). `news_relevance.py` now scores
each nearby headline with a cheap, dependency-light method: TF-IDF cosine
similarity against a small reference set of market-moving event archetypes
(earnings, M&A, litigation, guidance, a halt, an FDA result, etc.) plus a
keyword/entity boost, producing a 0-1 relevance score. A price jump is only
classified `explained_price_move` when a nearby headline clears
`config.py: news_relevance_explained_threshold` (0.45); a headline that
exists but doesn't clear it is still surfaced in the alert detail ("a
nearby headline scored only N% relevant") rather than silently ignored or
silently treated as an explanation. Both the synthetic demo path
(`alert_engine.build_synthetic_news_lookup`) and the live yfinance path
(`data/live_feed.build_live_news_lookup`, previously not wired into the
pipeline at all) now go through the same relevance model.

`stress_index.py` combines volatility regime (40%), correlation-break
severity (35%), and ensemble-flagged fraction of the watchlist (25%) into a
single 0-100 number — a hand-set, documented blend for legibility, not a
tuned trading signal.

### 5. Per-alert numeric confidence score (`alert_engine.py`)

**What changed in the review pass.** Every alert used to carry only a
3-level Low/Medium/High severity label — the brief's own example alert
format ("Anomaly Score: 94/100") has a single number, and this project's
only 0-100 figure used to be the watchlist-wide stress index, not a
per-alert one. Every `Alert` now also carries a `confidence: float` (0-100),
computed from a documented, simple blend already available at
alert-creation time: the ensemble's own normalized magnitude
(`models/ensemble.py`'s `ensemble_score`, which blends the Isolation
Forest's and autoencoder's z-scaled outputs) for ticker-level alerts, or the
correlation engine's `delta_zscore` for correlation alerts, squashed into
0-100 and saturating rather than hard-capping at any one value. Dual-model
consensus adds a fixed boost on top, so a consensus alert always scores at
or above the single-model baseline for the same evidence (verified by
`tests/test_pipeline.py::test_confidence_score_monotonic_and_consensus_boost`).
It's shown in the Alert Feed table, the detail panel, and the PDF incident
report header, right next to severity.

### 6. Background autorefresh (Live mode) (`app.py`, `services/pipeline_service.py`, `config.py`)

**What changed in the review pass — this was the single highest-leverage
fix in the whole review.** `load_live()`/`run_pipeline()` used to only
recompute on page load or an explicit "Refresh live data" click — nothing
re-pulled or re-scored on a timer, so "continuously watches" was aspirational
language, and there was no bounded, quotable detection-latency number ("it's
however long since the last click" is not a number).

**v2.2 update:** the original fix used the third-party
[`streamlit-autorefresh`](https://pypi.org/project/streamlit-autorefresh/)
package, which reran the *entire* script on a timer. That's now replaced
with Streamlit's own native `st.fragment(run_every=...)` (`streamlit>=1.37`):
the Command Center header and Alert Feed are each their own fragment,
refreshing on a sidebar-configurable 10/20/60s interval *without* rerunning
the whole page, so heavier tabs (3D landscape, order-flow simulation,
replay) don't recompute every tick. `load_live`'s own
`@st.cache_data(ttl=...)` still matches the interval, so a fragment tick is
a genuine re-pull of fresh bars, not a cosmetic re-render of stale cached
data. Live fetches also retry twice with exponential backoff
(`services/pipeline_service.py::_fetch_live_with_retry`) before falling
back to simulated, so one transient yfinance hiccup doesn't drop the whole
session out of Live mode; and a stale-data check compares the last
successful refresh against `stale_data_multiplier x` the configured
interval, degrading the green "continuous monitoring" banner to an orange
"data delayed" one instead of silently showing an old snapshot as current.
A closer-to-production next step (documented, not yet built — see *Known
limitations*) is a small in-process background thread/APScheduler job that
polls independently of whether a browser tab is open.

## Validating detection quality

Because the simulated generators inject **known** anomalies (both at the bar
level and the order-event level), detection can be checked quantitatively —
something you can't do against real, unlabeled market data:

- Bar-level recall on injected anomalies: **~95%** average across 10 random seeds
- Order-flow surveillance: **all 4 pattern types caught** in the test scenario
- False "High severity" alerts on a pure-noise (zero anomaly) scenario: **<10** out of 2,800 ticker-bars
- Correlated-group-move recall unchanged (**~87%** across 15 seeds) after adding the concentration-ratio check, and a
  uniform whole-watchlist shock is now correctly classified as `market_wide_move`, never `correlated_group_move`
  (`tests/test_pipeline.py::test_market_wide_move_not_flagged_as_cluster`)

This is a demo-scale sanity check, not a production accuracy claim — real
markets are far messier — but it's real evidence the pipeline does what it
says, which is a stronger pitch than "trust me."

### v2.2's own test plan, actually run (not just written down)

The v2.2 upgrade plan's own Section 32 lists exactly what to check before
calling any of it done, and says plainly not to claim completeness
untested. Running every row of that table for real — not just the parts
covered by re-running the pre-existing suite — caught two genuine bugs
that a code-read alone had missed:

- **System Health falsely reported "OK" on a failed live fetch.** Forcing
  a live-data failure (this sandbox's own network policy blocks outbound
  yfinance calls, which turned out to be a convenient real-world test
  case) showed the Data Source check silently passing instead of warning,
  because it inferred "was Live requested" from the pipeline's (by-then
  relabeled) `mode` field instead of its `error` field. Fixed in
  `services/health_service.py::check_data_source` to check `error`
  first, independent of how `mode` got relabeled on fallback.
- **A live fetch with no explicit timeout could hang past the refresh
  interval.** `yf.download()`'s own `timeout=` only bounds part of what it
  does, and `yf.Ticker(...).news` exposes no timeout at all — on a network
  that silently drops packets instead of actively rejecting them (this
  sandbox's proxy does the latter, so the gap wasn't visible until traced
  through by hand), either call can block for a full OS-level TCP timeout
  (60s+), during which the header/feed fragments appear frozen even
  though nothing crashed. Fixed by wrapping every outbound yfinance call
  in `data/live_feed.py` with a hard `CONFIG.live_fetch_timeout_s` (8s)
  ceiling via a small thread-pool helper, so a bad network fails fast and
  falls back, keeping "continuous monitoring" true under real-world flaky
  connectivity, not just a clean connection.

Also fixed along the way: the Command Center header had no visible
"last refreshed" indicator at all, so the plan's "watch it tick for 3
cycles" check had nothing to watch — added one line ("🕒 Header last
refreshed: HH:MM:SS") that the fragment updates every cycle; and the
investigation-workflow "Current status" caption lagged one click behind
its own save button (it read `stored` before the button's handler ran),
fixed with a `st.rerun()` after a successful save.

`tests/test_pipeline.py` was kept **byte-for-byte unmodified** throughout,
per the plan's own explicit requirement — including reverting an earlier,
unnecessary rename (`_confidence_score` → `_correlation_alert_score`) that
would otherwise have forced an edit to that file's import. The two new
automated checks the plan's Section 32 also calls for but the existing
suite doesn't cover (composite-score monotonicity, and an automated
persistence round-trip) live in a separate `tests/test_sprint_upgrades.py`
instead, specifically so this file's "must pass unmodified" guarantee is
never put at risk by a future edit here.

## Deploying on Render

This working copy includes `render.yaml`, `.streamlit/config.toml`
(which also sets `client.showErrorDetails = false`, so an unhandled
exception never leaks a full Python traceback to a visitor), and a
`.gitignore` -- verify with `ls -la .streamlit .gitignore render.yaml`
before you push. **A previous push of this project to a shared GitHub repo
was found (by an independent review) to be missing `.streamlit/config.toml`
and `.gitignore`** -- almost certainly because it was a partial "Add files
via upload" web upload rather than a `git push` of this exact folder, and
GitHub's upload UI can silently drop dotfiles/dot-directories if they
aren't explicitly selected. If you're pushing to an existing repo that
might be in that state, **replace its contents with this folder** rather
than adding to it, and confirm afterward with a fresh
`git clone` that the file list matches what's below:

1. Push this folder to a GitHub repo (`git init` has already been run
   locally with clean, descriptive commits — just add a remote and push):
   ```bash
   git remote add origin https://github.com/<you>/unusual-activity-watchdog.git
   git branch -M main
   git push -u origin main
   ```
2. On [render.com](https://render.com), **New +** -> **Blueprint**, connect
   the repo -- Render reads `render.yaml` and configures the web service
   automatically. For a plain **Web Service** instead:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
3. Deploy. First build takes a few minutes (installing scikit-learn/SHAP/etc).

No environment variables or API keys are required for either mode --
yfinance's public endpoints need no key, and the simulated demo path needs
no external service at all.

Render's free tier spins down after 15 min idle (cold start ~30-60s — open
the link a minute before your judging slot), and has unrestricted outbound
internet, so **Live/yfinance mode will actually reach real data there**,
unlike a locked-down sandbox. A single web service is all that's needed --
the Live-mode autorefresh (see architecture section 6 above) runs as an
in-process timer inside that same process, not a separate worker dyno.

## Known limitations & honest next steps

- **No real trade-tick / order-book feed.** Free data (yfinance) gives OHLCV
  bars, not individual trades. The order-flow surveillance tab is explicitly
  a simulated testbed for that reason — labeled honestly as such rather than
  implied to run on real order data, which no free source provides.
- **yfinance may be unreachable from a locked-down network** (it was, in the
  sandbox this was built in) — the app detects this and falls back to the
  simulated scenario automatically (`data_source.py::resolve_market_data`,
  unit-tested with a fake failing fetch so the fallback path itself is
  verified, not just manually inspected).
- **News relevance scoring is TF-IDF + keyword matching, not a full
  NLP/entity-resolution or sentiment stack.** `news_relevance.py` upgrades
  the check from bare presence to a 0-1 relevance score (see architecture
  section 4 above), which closes the literal gap the problem statement
  names, but it's still a lexical/statistical match, not true language
  understanding — a production system would want named-entity resolution
  and a proper sentiment model on top.
- **Anomaly scoring is a periodic retrospective sweep of the recent window,
  not literal tick-by-tick streaming inference.** `models/anomaly_model.py`
  and `models/autoencoder_model.py` each fit once on the whole per-ticker
  feature window, then score that same window — a later bar can shape what
  "normal" means for an earlier one in the same window. The input features
  themselves (rolling z-scores) are correctly causal, but this is closer to
  "a surveillance desk reviewing the last session" than literal streaming
  inference, and the app's own wording says so now rather than implying
  otherwise. A real walk-forward fix (refit every K bars on strictly-past
  data, score only the unseen tail) is a scoped, well-understood next step,
  staged for after this round given its cost relative to the wording fix.
- **The autorefresh is a client-driven timer, not an independent background
  process.** The native `st.fragment(run_every=...)` mechanism (see
  architecture section 6 above) makes "continuously watches" true while a
  browser tab is open; a from-scratch backend would add a small in-process
  poller thread (or APScheduler job) that keeps running independently of
  any open tab — documented as the next step up, not needed for free-tier
  Render (Phase 10 of the review this pass responded to: the whole process,
  poller included, stays alive between requests on Render's free tier as
  long as it isn't idle long enough to spin down).
- **Alert persistence (SQLite) survives the running instance only.** On a
  host with no persistent disk (e.g. Render's free tier), `data/store.py`'s
  SQLite file lives on that instance's local, ephemeral filesystem —
  investigation status/notes survive reruns and reconnects while the
  instance keeps running, but are wiped on every redeploy or cold restart.
  A production deployment would point it at a managed Postgres/SQLite-on-a-
  persistent-volume instead; the module's own functions (parameterized
  queries throughout) would port over with a connection-string change, not
  a rewrite.
- **The autoencoder is an MLP, not a recurrent/transformer model.** A
  deliberate speed/footprint tradeoff for free-tier hosting and instant
  demo response (see architecture section above) — swapping in an LSTM is a
  contained change to `models/autoencoder_model.py` if heavier compute is available.
- **Thresholds are hand-tuned for a 7-stock, 5-minute-bar demo watchlist,
  chosen by inspection against the injected ground truth, not a formal
  precision/recall sweep** — though the test harness to run that sweep
  already exists (`tests/test_pipeline.py`). `config.py` centralizes every
  threshold for a one-file retune.
- **Models retrain from scratch on each app refresh.** Fine at demo scale
  (sub-second per ticker); production would persist and incrementally
  update models instead of refitting on every page load.
- **No liquidity- or time-of-day-adjusted thresholds.** A thin, low-liquidity
  stock's normal volume noise could look like a burst under the same fixed
  z-score logic used for a large-cap, and the first/last bars of a session
  (typically noisier in real markets) are scored the same as midday bars.
  Not addressed in this pass -- noted honestly rather than silently ignored.
- **No authentication, rate limiting, or dependency vulnerability scan.**
  Fine for a public hackathon demo with no user data and no paid/keyed API;
  would be real gaps for a production surveillance desk tool. Run
  `pip-audit` once before submission as a cheap sanity check.

## Tech stack

Python, pandas, numpy, scikit-learn (Isolation Forest + MLP autoencoder +
TF-IDF for news relevance scoring), SHAP, Plotly, Streamlit (>=1.37, for
native `st.fragment`), yfinance, reportlab, matplotlib, and the standard
library's `sqlite3` for alert persistence. No environment variables or API
keys required for either data-source mode.
