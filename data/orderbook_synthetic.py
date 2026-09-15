"""
Synthetic limit-order-book EVENT stream generator.

Everything else in this project works at the OHLCV bar level (one price/
volume per 5 minutes), which is the finest grain free data gives you. Real
exchange surveillance desks (this problem statement is literally describing
their job) watch something finer: the actual stream of order adds, cancels,
and executions, because several classic manipulation patterns are only
visible there:

  - SPOOFING   -- place a big order to fake pressure, cancel before it fills
  - LAYERING   -- spoofing's bigger sibling: several fake orders stacked
                  across price levels at once
  - QUOTE STUFFING -- flood the book with adds/cancels far faster than any
                  human trader would, to jam competitors' systems
  - WASH TRADING   -- the same actor buys and sells the same size at the
                  same price in a short window, creating fake volume/
                  liquidity signals with no real change in ownership

No free data source hands you real per-order, per-trader event streams
(that's proprietary exchange data), so this module generates a synthetic
one with the same known-ground-truth-injection approach as
data/synthetic.py -- it's a labeled surveillance testbed, not a claim about
real order flow. Framed honestly, this is still a legitimate demonstration
of exchange-grade surveillance logic layered on top of the bar-level
anomaly detection everything else in this project does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_order_events(
    ticker: str,
    n_minutes: int = 120,
    events_per_minute: int = 8,
    n_traders: int = 40,
    mid_price: float = 200.0,
    seed: int = 3,
    inject_patterns: bool = True,
) -> dict:
    """Generate a synthetic order-event stream for one ticker.

    Returns {"events": DataFrame, "ground_truth": list[dict]}. Each event
    row: timestamp, trader_id, side (buy/sell), event (add/cancel/execute),
    price, size.
    """
    rng = np.random.default_rng(seed)
    n_events = n_minutes * events_per_minute
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n_events,
                                freq=pd.Timedelta(seconds=60 / events_per_minute))

    traders = [f"T{100 + i}" for i in range(n_traders)]
    events = []
    open_orders: dict[str, dict] = {}
    order_seq = 0

    def new_order_id():
        nonlocal order_seq
        order_seq += 1
        return f"{ticker}-{order_seq}"

    for ts in timestamps:
        # Ordinary background order flow: small orders, mostly filled or
        # cancelled at unremarkable rates -- this is the "normal" noise
        # floor the detectors have to see past.
        for _ in range(rng.poisson(1.1)):
            trader = rng.choice(traders)
            side = rng.choice(["buy", "sell"])
            price = round(mid_price + rng.normal(0, 0.15), 2)
            size = int(abs(rng.normal(150, 60))) + 10
            oid = new_order_id()
            events.append({"timestamp": ts, "order_id": oid, "trader_id": trader,
                            "side": side, "event": "add", "price": price, "size": size})
            open_orders[oid] = {"trader": trader, "side": side, "price": price, "size": size, "ts": ts}
            # most background orders resolve quickly (fill or ordinary cancel)
            if rng.random() < 0.6:
                resolution = "execute" if rng.random() < 0.7 else "cancel"
                events.append({"timestamp": ts, "order_id": oid, "trader_id": trader,
                                "side": side, "event": resolution, "price": price, "size": size})
                open_orders.pop(oid, None)

    ground_truth = []
    if inject_patterns:
        # --- Inject one of each pattern at well-separated times ---------
        pattern_starts = sorted(rng.choice(range(10, n_minutes - 15), size=4, replace=False))

        # 1) SPOOFING: one huge order far from touch, cancelled ~30s later
        t0 = pattern_starts[0]
        spoof_trader = rng.choice(traders)
        spoof_side = rng.choice(["buy", "sell"])
        spoof_price = round(mid_price + (0.6 if spoof_side == "buy" else -0.6), 2)
        spoof_size = int(rng.uniform(4000, 7000))
        ts_place = timestamps[t0 * events_per_minute]
        ts_cancel = ts_place + pd.Timedelta(seconds=25)
        oid = new_order_id()
        events.append({"timestamp": ts_place, "order_id": oid, "trader_id": spoof_trader,
                        "side": spoof_side, "event": "add", "price": spoof_price, "size": spoof_size})
        events.append({"timestamp": ts_cancel, "order_id": oid, "trader_id": spoof_trader,
                        "side": spoof_side, "event": "cancel", "price": spoof_price, "size": spoof_size})
        ground_truth.append({"kind": "spoofing", "trader_id": spoof_trader, "timestamp": ts_place,
                              "description": f"{spoof_trader} placed a {spoof_size}-share {spoof_side} order, "
                                              f"then cancelled it {25}s later without any fill"})

        # 2) LAYERING: same trader stacks 5 orders across price levels, all cancelled together
        t1 = pattern_starts[1]
        layer_trader = rng.choice(traders)
        layer_side = rng.choice(["buy", "sell"])
        ts_layer = timestamps[t1 * events_per_minute]
        layer_ids = []
        for level in range(5):
            offset = (level + 1) * 0.05 * (1 if layer_side == "buy" else -1)
            price = round(mid_price + offset, 2)
            size = int(rng.uniform(800, 1500))
            oid = new_order_id()
            layer_ids.append((oid, price, size))
            events.append({"timestamp": ts_layer, "order_id": oid, "trader_id": layer_trader,
                            "side": layer_side, "event": "add", "price": price, "size": size})
        ts_layer_cancel = ts_layer + pd.Timedelta(seconds=8)
        for oid, price, size in layer_ids:
            events.append({"timestamp": ts_layer_cancel, "order_id": oid, "trader_id": layer_trader,
                            "side": layer_side, "event": "cancel", "price": price, "size": size})
        ground_truth.append({"kind": "layering", "trader_id": layer_trader, "timestamp": ts_layer,
                              "description": f"{layer_trader} stacked {len(layer_ids)} {layer_side} orders "
                                              f"across price levels, all pulled within 8s -- classic fake-depth pattern"})

        # 3) QUOTE STUFFING: one trader fires dozens of add/cancel pairs in a few seconds
        t2 = pattern_starts[2]
        stuff_trader = rng.choice(traders)
        ts_stuff = timestamps[t2 * events_per_minute]
        n_stuff = 60
        for i in range(n_stuff):
            side = rng.choice(["buy", "sell"])
            price = round(mid_price + rng.normal(0, 0.05), 2)
            size = int(rng.uniform(50, 200))
            oid = new_order_id()
            t_add = ts_stuff + pd.Timedelta(milliseconds=i * 80)
            events.append({"timestamp": t_add, "order_id": oid, "trader_id": stuff_trader,
                            "side": side, "event": "add", "price": price, "size": size})
            events.append({"timestamp": t_add + pd.Timedelta(milliseconds=30), "order_id": oid, "trader_id": stuff_trader,
                            "side": side, "event": "cancel", "price": price, "size": size})
        ground_truth.append({"kind": "quote_stuffing", "trader_id": stuff_trader, "timestamp": ts_stuff,
                              "description": f"{stuff_trader} sent {n_stuff} add/cancel pairs in under 5 seconds -- "
                                              f"message-rate far beyond any plausible manual or normal-strategy trading"})

        # 4) WASH TRADING: same trader buys then sells the same size at the same price, moments apart
        t3 = pattern_starts[3]
        wash_trader = rng.choice(traders)
        ts_wash = timestamps[t3 * events_per_minute]
        wash_price = round(mid_price + rng.normal(0, 0.1), 2)
        wash_size = int(rng.uniform(500, 1200))
        oid1, oid2 = new_order_id(), new_order_id()
        events.append({"timestamp": ts_wash, "order_id": oid1, "trader_id": wash_trader,
                        "side": "buy", "event": "add", "price": wash_price, "size": wash_size})
        events.append({"timestamp": ts_wash, "order_id": oid1, "trader_id": wash_trader,
                        "side": "buy", "event": "execute", "price": wash_price, "size": wash_size})
        ts_wash2 = ts_wash + pd.Timedelta(seconds=4)
        events.append({"timestamp": ts_wash2, "order_id": oid2, "trader_id": wash_trader,
                        "side": "sell", "event": "add", "price": wash_price, "size": wash_size})
        events.append({"timestamp": ts_wash2, "order_id": oid2, "trader_id": wash_trader,
                        "side": "sell", "event": "execute", "price": wash_price, "size": wash_size})
        ground_truth.append({"kind": "wash_trading", "trader_id": wash_trader, "timestamp": ts_wash,
                              "description": f"{wash_trader} bought then sold {wash_size} shares at the same "
                                              f"{wash_price} price {4}s apart -- no net position change, just manufactured volume"})

    events_df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    return {"events": events_df, "ground_truth": ground_truth, "ticker": ticker}
