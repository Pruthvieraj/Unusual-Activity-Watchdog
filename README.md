# Unusual Activity Watchdog

**Synapse 1.0 (SIT Flagship Hackathon) — AI in FinTech track**

> Build a system that continuously watches live buying/selling activity for a
> stock and raises an alert when something breaks the normal pattern — a
> sudden burst of trades, a price jump with no news behind it, several stocks
> moving together in a suspicious way — so a human analyst can investigate.

This repo is a working prototype of exactly that: a Streamlit dashboard backed
by a three-signal anomaly-detection pipeline, explainable with SHAP, with a
built-in simulated market (so the demo works reliably regardless of market
hours, network access, or luck) and a real yfinance live-data path for
running it against actual markets.

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

Run the sanity-check test suite (checks the detector against known,
injected anomalies):

```bash
python3 tests/test_pipeline.py
```

## Deploying on Render

The repo already includes `render.yaml`, `.streamlit/config.toml`, and a
`.gitignore`, so it deploys as-is:

1. Push this folder to a GitHub repo (`git init` has already been run
   locally with one commit — just add a remote and push):
   ```bash
   git remote add origin https://github.com/<you>/unusual-activity-watchdog.git
   git branch -M main
   git push -u origin main
   ```
2. On [render.com](https://render.com), **New +** -> **Blueprint**, connect
   the repo -- Render reads `render.yaml` and configures the web service
   automatically (build command, start command, Python version). If you'd
   rather set it up as a plain **Web Service** instead of a Blueprint, use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
3. Deploy. First build takes a few minutes (installing scikit-learn/SHAP/etc).

Notes specific to Render's free tier:
- The service spins down after 15 minutes idle; the next visit takes ~30-60s
  to cold-start. Worth knowing if you're demoing live at a judging table --
  open the link a minute before your slot.
- Render's outbound internet is unrestricted, so **Live / historical
  (yfinance) mode will actually reach real market data here**, unlike a
  sandboxed dev environment that blocks it -- Simulated mode still works
  identically either way.

## Why three separate signals, not one model

The problem statement names three distinct patterns, and they genuinely need
different detection strategies — a single model can't see all three:

| Pattern | Why it needs its own approach |
|---|---|
| **Sudden burst of trades** | A per-stock question: is *this* volume unusual *for this stock*. A global model would just learn "big-cap = normal." |
| **Price jump with no news** | Two-part: is the move statistically unusual, *and* is there a news catalyst. Needs a price-anomaly signal AND a news lookup, fused together. |
| **Several stocks moving together suspiciously** | Invisible to any per-stock model by definition — each stock might look perfectly ordinary alone. Needs to look across the whole watchlist at once. |

## Architecture

```
data/synthetic.py     -- simulated multi-stock tape with injected, known anomalies
data/live_feed.py     -- real intraday data + news via yfinance
        |
        v
features.py           -- rolling z-scores (volume, return) + volatility, per ticker
        |
        v
models/anomaly_model.py    -- per-ticker Isolation Forest, SHAP-explained
correlation_watch.py        -- cross-stock rolling-correlation break detector
        |
        v
alert_engine.py        -- fuses both signals + news check into one ranked,
                           deduplicated alert feed
        |
        v
app.py                 -- Streamlit dashboard (Alert Feed / Price & Volume /
                           Correlation Monitor / Detection Accuracy tabs)
```

### 1. Per-ticker anomaly detection (`models/anomaly_model.py`)

An **Isolation Forest is trained separately for each ticker** on three
rolling features (`return_zscore`, `volume_zscore`, `volatility`) — not
pooled across the watchlist, because different stocks trade at very
different scales and a pooled model would just learn "which stock is this"
rather than "is this unusual for this stock."

Isolation Forest's `contamination` parameter guarantees it flags a fixed ~5%
of bars as "relatively" unusual *even on pure noise* — that's inherent to how
the algorithm ranks points, not a bug. To avoid alert fatigue from that noise
floor, every flagged point is also gated against an **absolute** z-score
threshold (`config.py`: `volume_zscore_alert`, `price_jump_sigma_alert`)
before it becomes a real alert — so an alert fires only when a move is both
*relatively* unusual for that stock AND *absolutely* large by a human-legible
standard.

Every flagged point is explained with **SHAP** (`TreeExplainer`), so each
alert says *which feature drove it* (e.g. "volume_zscore contributed most"),
not just "this looks weird."

### 2. Cross-stock correlation-break detection (`correlation_watch.py`)

At each bar, a short rolling pairwise-correlation matrix (recent behavior) is
compared against a much longer trailing baseline (normal co-movement). Two
design choices that made this actually work in testing:

- **Top-k pairwise deltas, not the average across all pairs.** Averaging
  across every pair in the watchlist dilutes a genuine 3-stock cluster into
  noise (only 3 of e.g. 21 pairs actually move). Looking at the top 3
  pairwise deltas keys directly onto "a small group moving together,"
  which is the actual pattern being hunted for.
- **Expanding-history z-score, not a short rolling z-score.** A short
  rolling baseline would include the anomalous bars themselves, dragging its
  own "normal" up and masking the event. An expanding (whole-history) mean/
  std barely moves for one localized event, so the anomaly still stands out.

Validated on the simulated generator: **~87% of injected correlated-group
events detected with the correct tickers named**, across 15 random seeds
(see `tests/test_pipeline.py`).

### 3. News check (`data/live_feed.py` / synthetic equivalent)

Uses yfinance's built-in `Ticker.news` (no separate news API or key needed)
to check whether a headline exists for a ticker within a configurable
lookback window (`config.py`: `news_lookback_hours`) around a flagged price
move — separating "explained" (news-driven, low priority) from "unexplained"
(high priority) jumps. In simulated mode, an equivalent synthetic news table
plays the same role, with "explained" moves deliberately given a matching
headline and injected anomalies deliberately withheld one.

### 4. Alert fusion (`alert_engine.py`)

Combines all three signals into one ranked feed, then **merges consecutive
alerts of the same kind/ticker(s)** into a single event spanning
`[first_seen, last_seen]` — a sustained volume burst or correlated move
otherwise trips the detector on every bar it lasts for, flooding the feed
with near-duplicates of the same event.

## Validating detection quality

Because the simulated generator injects a **known** set of anomalies with
ground truth, detection can be checked quantitatively — something you can't
do against real, unlabeled market data. The "Detection Accuracy" tab in the
app shows this live; `tests/test_pipeline.py` checks it automatically:

- Recall on injected anomalies: **~95%** average across 10 random seeds
- False "High severity" alerts on a pure-noise (zero anomaly) scenario: **<10** out of 2,800 ticker-bars

This is a demo-scale sanity check, not a production accuracy claim — real
markets are far messier than the synthetic generator — but it's real evidence
the pipeline does what it says, which is a stronger pitch than "trust me."

## Known limitations & honest next steps

- **No real trade-tick / order-book feed.** Free data (yfinance) gives OHLCV
  bars, not individual trades — "sudden burst of trades" is approximated as a
  volume burst per bar, which is the standard proxy without a paid feed
  (e.g. a market-data vendor's tick API).
- **yfinance may be unreachable from a locked-down network** (it was, in the
  sandbox this was built in) — the app detects this and falls back to the
  simulated scenario automatically, but on your own laptop/hotspot it should
  reach Yahoo Finance fine.
- **News matching is presence-based, not causal.** It checks "is there a
  headline nearby," not "does this headline actually explain this move" —
  a real system would want NLP sentiment/relevance scoring on the headline
  (spaCy/NLTK would drop in easily for this).
- **Thresholds are hand-tuned for a 7-stock, 5-minute-bar demo watchlist.**
  `config.py` centralizes every threshold so this is a one-file retune for a
  different watchlist or bar frequency, but a production system would want
  these learned/calibrated rather than fixed constants.
- **Isolation Forest is retrained from scratch on each app refresh.** Fine at
  demo scale (fits in well under a second per ticker); a production version
  would persist and incrementally update models instead of refitting on
  every page load.

## Tech stack

Python, pandas, numpy, scikit-learn (Isolation Forest), SHAP, Plotly,
Streamlit, yfinance.
