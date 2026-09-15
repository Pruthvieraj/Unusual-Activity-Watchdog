# Unusual Activity Watchdog

**Synapse 1.0 (SIT Flagship Hackathon) — AI in FinTech track**

> Build a system that continuously watches live buying/selling activity for a
> stock and raises an alert when something breaks the normal pattern — a
> sudden burst of trades, a price jump with no news behind it, several stocks
> moving together in a suspicious way — so a human analyst can investigate.

This repo is a working prototype of exactly that, built up into a full
surveillance platform: a **dual-AI ensemble** (Isolation Forest + neural
autoencoder) for per-stock anomalies, a **cross-stock correlation-break
engine**, exchange-grade **order-flow manipulation surveillance** (spoofing /
layering / quote stuffing / wash trading), a **live animated replay mode**
with a composite market-stress gauge, a **3D anomaly landscape**, a
**correlation network diagram**, browser **voice alerts**, and one-click
**PDF incident reports** — all wrapped in a Streamlit dashboard with a
built-in simulated market (so the demo works reliably regardless of market
hours, network access, or luck) plus a real yfinance live-data path.

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

Run the sanity-check test suite (checks every detector against known,
injected ground truth):

```bash
python3 tests/test_pipeline.py
```

## What's in the dashboard

| Tab | What it shows |
|---|---|
| **Alert Feed** | Ranked, deduplicated alerts with AI-agreement badges, SHAP driver charts, and a one-click PDF incident report per alert |
| **Price & Volume** | Full per-ticker detail view: price, volume, return z-score with alert thresholds drawn in |
| **Correlation Monitor** | Rolling correlation vs. baseline over time, plus current-vs-baseline correlation heatmaps |
| **3D & Network** | A live correlation network diagram (flagged clusters highlighted) and a 3D scatter of every bar in anomaly-feature space |
| **Live Replay** | Scrubs/auto-plays through the scenario bar-by-bar, revealing alerts and a market-stress gauge only up to the current point in time -- watch detection happen instead of reading a static table |
| **Order-Flow Surveillance** | Simulated order-event stream + rule-based detectors for spoofing, layering, quote stuffing, and wash trading |
| **Detection Accuracy** | Live recall check against the simulated scenario's known, injected ground truth |

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
data/live_feed.py          -- real intraday data + news via yfinance
data/orderbook_synthetic.py -- simulated order-EVENT stream (spoofing/layering/stuffing/wash trades)
        |
        v
features.py                 -- rolling z-scores (volume, return) + volatility, per ticker
        |
        v
models/anomaly_model.py     -- per-ticker Isolation Forest, SHAP-explained
models/autoencoder_model.py -- per-ticker neural autoencoder (reconstruction-error anomaly score)
models/ensemble.py          -- combines both into one score + a "dual-model consensus" flag
correlation_watch.py         -- cross-stock rolling-correlation break detector
order_flow.py                -- rule-based spoofing/layering/stuffing/wash-trade detectors
stress_index.py              -- composite 0-100 market-stress index
        |
        v
alert_engine.py              -- fuses all signals + news check into one ranked, deduplicated feed
report_generator.py          -- one-click PDF incident report per alert
viz_extra.py                 -- 3D anomaly landscape, correlation network, voice-alert snippet
        |
        v
app.py                       -- Streamlit dashboard (7 tabs, see above)
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

Validated on the simulated generator: **~87% of injected correlated-group
events detected with the correct tickers named**, across 15 random seeds.

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

`stress_index.py` combines volatility regime (40%), correlation-break
severity (35%), and ensemble-flagged fraction of the watchlist (25%) into a
single 0-100 number — a hand-set, documented blend for legibility, not a
tuned trading signal.

## Validating detection quality

Because the simulated generators inject **known** anomalies (both at the bar
level and the order-event level), detection can be checked quantitatively —
something you can't do against real, unlabeled market data:

- Bar-level recall on injected anomalies: **~95%** average across 10 random seeds
- Order-flow surveillance: **all 4 pattern types caught** in the test scenario
- False "High severity" alerts on a pure-noise (zero anomaly) scenario: **<10** out of 2,800 ticker-bars

This is a demo-scale sanity check, not a production accuracy claim — real
markets are far messier — but it's real evidence the pipeline does what it
says, which is a stronger pitch than "trust me."

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
   automatically. For a plain **Web Service** instead:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
3. Deploy. First build takes a few minutes (installing scikit-learn/SHAP/etc).

Render's free tier spins down after 15 min idle (cold start ~30-60s — open
the link a minute before your judging slot), and has unrestricted outbound
internet, so **Live/yfinance mode will actually reach real data there**,
unlike a locked-down sandbox.

## Known limitations & honest next steps

- **No real trade-tick / order-book feed.** Free data (yfinance) gives OHLCV
  bars, not individual trades. The order-flow surveillance tab is explicitly
  a simulated testbed for that reason — labeled honestly as such rather than
  implied to run on real order data, which no free source provides.
- **yfinance may be unreachable from a locked-down network** (it was, in the
  sandbox this was built in) — the app detects this and falls back to the
  simulated scenario automatically.
- **News matching is presence-based, not causal.** It checks "is there a
  headline nearby," not "does this headline actually explain this move" —
  a real system would want NLP sentiment/relevance scoring on the headline.
- **The autoencoder is an MLP, not a recurrent/transformer model.** A
  deliberate speed/footprint tradeoff for free-tier hosting and instant
  demo response (see architecture section above) — swapping in an LSTM is a
  contained change to `models/autoencoder_model.py` if heavier compute is available.
- **Thresholds are hand-tuned for a 7-stock, 5-minute-bar demo watchlist.**
  `config.py` centralizes every threshold for a one-file retune.
- **Models retrain from scratch on each app refresh.** Fine at demo scale
  (sub-second per ticker); production would persist and incrementally
  update models instead of refitting on every page load.

## Tech stack

Python, pandas, numpy, scikit-learn (Isolation Forest + MLP autoencoder),
SHAP, Plotly, Streamlit, yfinance, reportlab, matplotlib.
