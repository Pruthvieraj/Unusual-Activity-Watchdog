"""
Rule-based market-manipulation surveillance over an order-event stream.

These are the same heuristic shapes real exchange surveillance systems use
as their first-line detectors (before anything statistical/ML gets layered
on top): a large order that vanishes right before it would fill, several
orders stacked and pulled together, an inhuman message rate, or a trade
that nets out to nothing. None of this needs machine learning to be a real
signal -- it needs precise bookkeeping of order lifecycles, which is what
this module does.
"""

from __future__ import annotations

import pandas as pd

# --- tunable thresholds ----------------------------------------------------
SPOOF_SIZE_MULTIPLIER = 8.0        # order must be >= this x the trader's median size
SPOOF_MAX_LIFETIME_SEC = 45        # cancelled within this long to count as "fast"
LAYERING_MIN_LEVELS = 3            # distinct price levels from one trader, one side
LAYERING_WINDOW_SEC = 15           # ...within this window, all later cancelled
STUFFING_WINDOW_SEC = 5
STUFFING_MIN_EVENTS = 25           # add+cancel events from one trader in the window
WASH_PRICE_TOLERANCE = 0.02
WASH_WINDOW_SEC = 10


def detect_spoofing(events: pd.DataFrame) -> list[dict]:
    adds = events[events["event"] == "add"]
    cancels = events[events["event"] == "cancel"][["order_id", "timestamp"]].rename(columns={"timestamp": "cancel_ts"})
    executes = set(events.loc[events["event"] == "execute", "order_id"])

    merged = adds.merge(cancels, on="order_id", how="inner")
    merged = merged[~merged["order_id"].isin(executes)]  # only orders that never filled at all

    median_size_by_trader = events[events["event"] == "add"].groupby("trader_id")["size"].median()

    flags = []
    for _, row in merged.iterrows():
        lifetime = (row["cancel_ts"] - row["timestamp"]).total_seconds()
        baseline = median_size_by_trader.get(row["trader_id"], row["size"])
        if baseline <= 0:
            continue
        if row["size"] >= SPOOF_SIZE_MULTIPLIER * baseline and lifetime <= SPOOF_MAX_LIFETIME_SEC:
            flags.append({
                "kind": "spoofing", "timestamp": row["timestamp"], "trader_id": row["trader_id"],
                "severity": "High",
                "detail": (f"Order for {int(row['size']):,} shares ({row['size']/baseline:.1f}x this trader's "
                            f"typical size) on the {row['side']} side, cancelled after {lifetime:.0f}s with zero fill."),
            })
    return flags


def detect_layering(events: pd.DataFrame) -> list[dict]:
    adds = events[events["event"] == "add"].copy()
    flags = []
    for (trader, side), group in adds.groupby(["trader_id", "side"]):
        group = group.sort_values("timestamp")
        window = pd.Timedelta(seconds=LAYERING_WINDOW_SEC)
        # sliding window over this trader/side's adds
        times = group["timestamp"].values
        for i in range(len(group)):
            start = group["timestamp"].iloc[i]
            in_window = group[(group["timestamp"] >= start) & (group["timestamp"] <= start + window)]
            distinct_prices = in_window["price"].nunique()
            if distinct_prices >= LAYERING_MIN_LEVELS:
                order_ids = set(in_window["order_id"])
                cancels = events[(events["event"] == "cancel") & (events["order_id"].isin(order_ids))]
                executes = events[(events["event"] == "execute") & (events["order_id"].isin(order_ids))]
                if len(cancels) >= LAYERING_MIN_LEVELS and len(executes) == 0:
                    flags.append({
                        "kind": "layering", "timestamp": start, "trader_id": trader,
                        "severity": "High",
                        "detail": (f"{distinct_prices} {side} orders stacked across price levels within "
                                    f"{LAYERING_WINDOW_SEC}s, all cancelled with zero fills -- manufactured book depth."),
                    })
                    break  # one flag per trader/side cluster is enough
    return flags


def detect_quote_stuffing(events: pd.DataFrame) -> list[dict]:
    flags = []
    window = pd.Timedelta(seconds=STUFFING_WINDOW_SEC)
    for trader, group in events.groupby("trader_id"):
        group = group.sort_values("timestamp")
        ts = group["timestamp"]
        counts = ts.apply(lambda t: ((ts >= t) & (ts <= t + window)).sum())
        peak = counts.max()
        if peak >= STUFFING_MIN_EVENTS:
            idx = counts.idxmax()
            flags.append({
                "kind": "quote_stuffing", "timestamp": group.loc[idx, "timestamp"], "trader_id": trader,
                "severity": "Medium",
                "detail": f"{int(peak)} order messages from this trader within {STUFFING_WINDOW_SEC}s -- "
                          f"far above any plausible manual trading rate.",
            })
    return flags


def detect_wash_trading(events: pd.DataFrame) -> list[dict]:
    executes = events[events["event"] == "execute"].sort_values("timestamp")
    flags = []
    seen_pairs = set()
    window = pd.Timedelta(seconds=WASH_WINDOW_SEC)

    for trader, group in executes.groupby("trader_id"):
        buys = group[group["side"] == "buy"]
        sells = group[group["side"] == "sell"]
        for _, b in buys.iterrows():
            candidates = sells[
                (sells["timestamp"] >= b["timestamp"] - window) &
                (sells["timestamp"] <= b["timestamp"] + window) &
                (abs(sells["price"] - b["price"]) <= WASH_PRICE_TOLERANCE) &
                (sells["size"] == b["size"])
            ]
            for _, s in candidates.iterrows():
                pair_key = tuple(sorted([b["order_id"], s["order_id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                flags.append({
                    "kind": "wash_trading", "timestamp": min(b["timestamp"], s["timestamp"]), "trader_id": trader,
                    "severity": "High",
                    "detail": (f"Bought and sold {int(b['size']):,} shares at {b['price']} within "
                                f"{abs((b['timestamp']-s['timestamp']).total_seconds()):.0f}s -- no net position change."),
                })
    return flags


def run_surveillance(events: pd.DataFrame) -> pd.DataFrame:
    """Run all four detectors and return one ranked DataFrame."""
    flags = (
        detect_spoofing(events) + detect_layering(events) +
        detect_quote_stuffing(events) + detect_wash_trading(events)
    )
    if not flags:
        return pd.DataFrame(columns=["timestamp", "kind", "trader_id", "severity", "detail"])
    df = pd.DataFrame(flags)
    severity_rank = {"High": 0, "Medium": 1, "Low": 2}
    df["_rank"] = df["severity"].map(severity_rank)
    return df.sort_values(["timestamp", "_rank"]).drop(columns="_rank").reset_index(drop=True)
