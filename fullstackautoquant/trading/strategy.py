import argparse
import json
import math
import os

import numpy as np

try:
    from .utils import (
        compute_allowed_price,
        compute_limit_price_from_rt_preclose,
        compute_manual_price,
        ensure_logs_dir,
        fetch_tushare_quotes,
        gm_to_instrument,
        instrument_to_gm,
        load_config,
        min_order_lot_for_symbol,
        read_close_prices_from_h5,
        round_price,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    from pathlib import Path

    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from utils import (  # type: ignore
        compute_allowed_price,
        compute_limit_price_from_rt_preclose,
        compute_manual_price,
        ensure_logs_dir,
        fetch_tushare_quotes,
        instrument_to_gm,
        load_config,
        min_order_lot_for_symbol,
        read_close_prices_from_h5,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Strategy: dynamic rebalance to targets and orders")
    p.add_argument("--signals", required=True, help="signals JSON path")
    p.add_argument("--risk_state", required=False, default=None, help="risk_state JSON path")
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--targets", required=False, default=None, help="output targets json path")
    p.add_argument("--orders", required=False, default=None, help="output orders json path")
    p.add_argument("--current_positions", required=False, default=None, help="optional current positions json path")
    p.add_argument("--capital_override", required=False, default=None, help="override total capital(float)")
    p.add_argument("--account-id", dest="account_id", required=False, default=None, help="override GM account id for fetching positions")
    return p.parse_args()


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_current_positions_from_gm(cfg: dict, account_id_override: str | None = None) -> dict[str, int]:
    """Fetch current positions from gmtrade; return {GM symbol: shares}. Empty on failure."""
    try:
        from gm_api_wrapper import gm_login  # type: ignore
        try:
            from gmtrade.api import get_positions  # type: ignore
        except Exception:  # noqa: E722
            return {}
        # login via shared helper (reads .env or config)
        try:
            gm_login(cfg, account_id_override=account_id_override)
        except TypeError:
            # backward compat for gm_login(cfg)
            gm_login(cfg)
        positions = get_positions()
        cur: dict[str, int] = {}
        for p in positions or []:  # type: ignore
            sym = None
            if hasattr(p, "symbol"):
                sym = getattr(p, "symbol", None)
            elif isinstance(p, dict):
                sym = p.get("symbol")
            sh = None
            for fn in [
                "volume", "qty", "current_qty", "position", "amount", "shares", "quantity", "current_amount",
            ]:
                if hasattr(p, fn):
                    try:
                        sh = int(float(getattr(p, fn)))
                        break
                    except Exception:  # noqa: E722
                        pass
                if isinstance(p, dict) and fn in p:
                    try:
                        sh = int(float(p.get(fn) or 0))
                        break
                    except Exception:  # noqa: E722
                        pass
            if isinstance(sym, str) and sh and sh > 0:
                cur[sym.upper()] = int(sh)
        return cur
    except Exception:  # noqa: E722
        return {}


def infer_reference_prices(cfg: dict, qlib_instruments: list[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    price_src = cfg["order"].get("price_source", "qlib_close")
    if price_src == "tushare":
        gm_symbols: list[str] = []
        for ins in qlib_instruments:
            gm = instrument_to_gm(ins)
            if gm:
                gm_symbols.append(gm)
        quotes = fetch_tushare_quotes(gm_symbols, src=cfg["order"].get("tushare_src", "sina"))
        res: dict[str, float] = {}
        for gm_symbol in gm_symbols:
            info = quotes.get(gm_symbol, {})
            price = float(info.get("price") or info.get("ask") or info.get("bid") or info.get("pre_close") or 0.0)
            if price > 0:
                res[gm_symbol] = price
        return res, quotes
    h5_path = cfg["paths"]["daily_pv"]
    closes = read_close_prices_from_h5(h5_path, qlib_instruments)
    return closes, {}


def waterfill_with_cap(base_weights: list[float], max_w: float) -> list[float]:
    """Water-filling normalization with per-name cap.
    - base_weights non-negative; we normalize to sum 1 without exceeding max_w per name.
    - If all names hit cap before sum reaches 1, leftover remains as cash (sum<1).
    """
    if not base_weights:
        return []
    bw = np.array(base_weights, dtype=float)
    if np.sum(bw) <= 0:
        # fallback to equal
        bw = np.ones_like(bw)
    bw = bw / np.sum(bw)
    w = np.minimum(bw, max_w)
    locked = w >= max_w - 1e-12
    leftover = 1.0 - float(np.sum(w))
    # iteratively distribute leftover to unlocked indices
    for _ in range(10 * len(bw)):
        if leftover <= 1e-12:
            break
        free_idx = np.where(~locked)[0]
        if len(free_idx) == 0:
            break
        bw_free_sum = float(np.sum(bw[free_idx]))
        if bw_free_sum <= 1e-15:
            break
        # proposed add per free name
        add = leftover * (bw[free_idx] / bw_free_sum)
        # cap by (max_w - w)
        room = max_w - w[free_idx]
        add_clamped = np.minimum(add, room)
        w[free_idx] += add_clamped
        locked = w >= max_w - 1e-12
        leftover = 1.0 - float(np.sum(w))
    return w.tolist()


def _resolve_weight_mode(cfg: dict) -> tuple[str, dict]:
    weight_cfg = cfg.get("weights") or {}
    if not isinstance(weight_cfg, dict):
        weight_cfg = {}
    portfolio_cfg = cfg.get("portfolio") or {}
    if not isinstance(portfolio_cfg, dict):
        portfolio_cfg = {}
    raw_mode = weight_cfg.get("mode") or portfolio_cfg.get("weight_mode") or "equal"
    mode = str(raw_mode).strip().lower()
    if mode not in {"equal", "ranked"}:
        mode = "equal"
    return mode, weight_cfg


def _ranked_weight_candidates(signals_top: list[dict], weight_cfg: dict) -> np.ndarray:
    n = len(signals_top)
    if n == 0:
        return np.array([], dtype=float)
    metric = str(weight_cfg.get("rank_metric") or weight_cfg.get("metric") or "rank").strip().lower()
    exponent = weight_cfg.get("rank_exponent", weight_cfg.get("rank_power", 1.0))
    try:
        exponent_val = float(exponent)
    except (TypeError, ValueError):
        exponent_val = 1.0
    exponent_val = max(exponent_val, 0.0)
    if metric == "score":
        values = np.array([float(s.get("score", 0.0)) for s in signals_top], dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        span = float(values.max() - values.min())
        if not math.isfinite(span) or span <= 1e-12:
            metric = "rank"
        else:
            values = values - values.min()
            values = values + 1e-6
    if metric != "score":
        values = np.arange(n, 0, -1, dtype=float)
    if exponent_val != 1.0:
        values = np.power(values, exponent_val)
    return values


def compute_weight_candidates(signals_top: list[dict], cfg: dict) -> list[float]:
    if not signals_top:
        return []
    mode, weight_cfg = _resolve_weight_mode(cfg)
    if mode == "ranked":
        weights = _ranked_weight_candidates(signals_top, weight_cfg)
    else:
        weights = np.ones(len(signals_top), dtype=float)

    confidence_tilt = weight_cfg.get("confidence_tilt")
    if confidence_tilt is None:
        confidence_tilt = (mode == "equal")
    elif isinstance(confidence_tilt, str):
        confidence_tilt = confidence_tilt.strip().lower() in {"1", "true", "yes", "on"}
    else:
        confidence_tilt = bool(confidence_tilt)
    if confidence_tilt:
        conf = np.array([float(s.get("confidence", 1.0)) for s in signals_top], dtype=float)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
        tilt = 0.5 + 0.5 * np.clip(conf, 0.0, 1.0)
        weights = weights * tilt

    weights = np.clip(weights, 0.0, None)
    if np.sum(weights) <= 1e-12:
        weights = np.ones(len(signals_top), dtype=float)
    return weights.tolist()


def build_targets(
    signals: list[dict],
    cfg: dict,
    allow_buy: bool,
    limit_up_syms: list[str],
    limit_down_syms: list[str],
    total_capital: float,
    rank_map: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict], dict[str, float], dict[str, dict[str, float]], float, float]:
    topk = int(cfg["portfolio"]["topk"])
    invest_ratio = float(cfg["portfolio"]["invest_ratio"])
    max_w = float(cfg["portfolio"]["max_weight"])
    default_lot = int(cfg["portfolio"]["lot"])

    buy_offset = float(cfg["order"]["buy_limit_offset"])
    sell_offset = float(cfg["order"]["sell_limit_offset"])
    limit_threshold = float(cfg["order"]["limit_threshold"])
    tick = float(cfg["order"]["clamp_tick"])

    signals_sorted = sorted(signals, key=lambda x: (x.get("score", 0.0), x.get("confidence", 0.0)), reverse=True)
    signals_top = signals_sorted[:topk]

    price_source = cfg["order"].get("price_source", "qlib_close")
    qlib_instruments: list[str] = []
    for s in signals_top:
        ins = s.get("instrument")
        if isinstance(ins, str):
            qlib_instruments.append(ins)
    ref_prices_map, quote_details = infer_reference_prices(cfg, qlib_instruments)

    invest_cap = total_capital * invest_ratio

    targets: list[dict] = []
    orders: list[dict] = []
    provisional_buys: list[dict] = []

    n = len(signals_top)
    if n == 0:
        return targets, orders, ref_prices_map, quote_details, invest_cap, 0.0

    raw_weights = compute_weight_candidates(signals_top, cfg)
    w_final = waterfill_with_cap(raw_weights, max_w=max_w)

    total_target_value = 0.0
    for s, w in zip(signals_top, w_final, strict=False):
        symbol = s["symbol"]
        ins = s["instrument"]
        ref = float(ref_prices_map.get(symbol, 0.0))
        if ref <= 0:
            continue
        target_amt = invest_cap * w

        if cfg.get("order", {}).get("mode", "auto") == "manual":
            limit_buy = compute_manual_price("BUY", ref, buy_offset, sell_offset, tick)
        else:
            if price_source == "tushare":
                quote = quote_details.get(symbol, {})
                rt_price = float(quote.get("ask") or quote.get("price") or quote.get("bid") or ref)
                pre_close = float(quote.get("pre_close") or ref)
                if rt_price > 0 and pre_close > 0:
                    limit_buy = compute_limit_price_from_rt_preclose(
                        "BUY",
                        rt_price,
                        pre_close,
                        buy_offset,
                        sell_offset,
                        limit_threshold,
                        tick,
                    )
                else:
                    limit_buy = compute_allowed_price("BUY", ref, buy_offset, sell_offset, limit_threshold, tick)
            else:
                limit_buy = compute_allowed_price("BUY", ref, buy_offset, sell_offset, limit_threshold, tick)

        lot = max(default_lot, min_order_lot_for_symbol(symbol))
        lot_value = limit_buy * lot
        lots = int(target_amt // lot_value)
        if lots <= 0:
            continue
        target_shares = lots * lot
        max_amt = invest_cap * max_w
        if target_shares * limit_buy > max_amt + 1e-6:
            max_lots = int(max_amt // lot_value)
            target_shares = max_lots * lot
            if target_shares <= 0:
                continue

        display_price = ref
        if price_source == "tushare":
            quote = quote_details.get(symbol, {})
            display_price = float(quote.get("price") or quote.get("ask") or quote.get("bid") or ref)
            if display_price > 0:
                ref_prices_map[symbol] = display_price
            if quote:
                quote_details.setdefault(symbol, {}).update(quote)

        targets.append({
            "symbol": symbol,
            "instrument": ins,
            "target_shares": int(target_shares),
            "weight": float(w),
            "ref_price": float(display_price),
        })
        total_target_value += float(display_price) * int(target_shares)
        if allow_buy:
            est_cost = float(limit_buy) * int(target_shares)
            buy_volume = int(target_shares)
            if cfg.get("order", {}).get("mode", "auto") == "manual":
                buy_volume = max(lot, int(target_shares * 0.4) // lot * lot)
            provisional_buys.append({
                "symbol": symbol,
                "side": "BUY",
                "volume": int(buy_volume),
                "price": float(limit_buy),
                "type": "limit",
                "est_cost": float(limit_buy) * int(buy_volume),
            })

    budget = invest_cap
    rank_lookup = rank_map or {}
    provisional_buys = sorted(
        provisional_buys,
        key=lambda x: (
            -rank_lookup.get(str(x.get("symbol")), 10**9),
            -(x.get("est_cost", 0.0) or 0.0),
        ),
    )
    order_cfg = cfg.get("order", {})
    min_buy_cash = float(order_cfg.get("min_buy_cash") or order_cfg.get("min_buy_notional") or 0.0)
    for od in provisional_buys:
        est_cost = float(od.get("est_cost", 0.0) or 0.0)
        if min_buy_cash > 0 and est_cost < min_buy_cash:
            continue
        if est_cost <= budget + 1e-6:
            orders.append(od)
            budget -= est_cost
        else:
            lot = max(default_lot, min_order_lot_for_symbol(od["symbol"]))
            unit_cost = float(od.get("price", 0.0) or 0.0) * lot
            if unit_cost <= 0:
                continue
            max_lots = int(budget // unit_cost)
            if max_lots <= 0:
                continue
            adjusted_volume = max_lots * lot
            est_cost_adjusted = float(od.get("price", 0.0) or 0.0) * adjusted_volume
            if min_buy_cash > 0 and est_cost_adjusted < min_buy_cash:
                continue
            orders.append({
                **od,
                "volume": adjusted_volume,
                "est_cost": est_cost_adjusted,
            })
            budget -= est_cost_adjusted

    return targets, orders, ref_prices_map, quote_details, invest_cap, total_target_value


def dynamic_adjust_with_positions(
    targets: list[dict],
    current_positions: dict[str, int],
    cfg: dict,
    ref_prices: dict[str, float],
    quote_details: dict[str, dict[str, float]],
    price_source: str,
    allow_buy: bool,
    limit_up_syms: list[str],
    limit_down_syms: list[str],
) -> list[dict]:
    lot = int(cfg["portfolio"]["lot"])
    n_drop = int(cfg["portfolio"]["n_drop"])
    topk = int(cfg["portfolio"]["topk"])

    order_cfg = cfg.get("order", {})
    min_buy_cash = float(order_cfg.get("min_buy_cash") or order_cfg.get("min_buy_notional") or 0.0)
    min_sell_cash = float(order_cfg.get("min_sell_cash") or order_cfg.get("min_sell_notional") or 0.0)

    buy_offset = float(cfg["order"]["buy_limit_offset"])
    sell_offset = float(cfg["order"]["sell_limit_offset"])
    limit_threshold = float(cfg["order"]["limit_threshold"])
    tick = float(cfg["order"]["clamp_tick"])

    target_map = {t["symbol"]: int(t["target_shares"]) for t in targets}

    sells_target: list[tuple[str, int]] = []
    sells_full_exits: list[tuple[str, int]] = []

    for sym, cur in current_positions.items():
        cur = int(cur)
        if sym in target_map:
            tgt = int(target_map.get(sym, 0))
            if tgt < cur:
                volume = ((cur - tgt) // lot) * lot
                if volume > 0 and sym not in set(limit_down_syms):
                    sells_target.append((sym, volume))
        else:
            if cur > 0 and sym not in set(limit_down_syms):
                volume = (cur // lot) * lot
                if volume > 0:
                    sells_full_exits.append((sym, volume))

    sells_full_exits = sells_full_exits[:n_drop]

    current_symbols_count = sum(1 for sh in current_positions.values() if sh > 0)
    num_full_exits = len(sells_full_exits)
    allowed_new_slots = max(0, topk - (current_symbols_count - num_full_exits))

    orders: list[dict] = []

    for sym, vol in sells_target + sells_full_exits:
        ref = ref_prices.get(sym)
        if ref is None or ref <= 0:
            continue
        if cfg.get("order", {}).get("mode", "auto") == "manual":
            price = compute_manual_price("SELL", ref, buy_offset, sell_offset, tick)
        else:
            if price_source == "tushare":
                quote = quote_details.get(sym, {})
                rt_price = float(quote.get("bid") or quote.get("price") or ref)
                pre_close = float(quote.get("pre_close") or ref)
                if rt_price > 0 and pre_close > 0:
                    price = compute_limit_price_from_rt_preclose(
                        "SELL",
                        rt_price,
                        pre_close,
                        buy_offset,
                        sell_offset,
                        limit_threshold,
                        tick,
                    )
                else:
                    price = compute_allowed_price("SELL", ref, buy_offset, sell_offset, limit_threshold, tick)
            else:
                price = compute_allowed_price("SELL", ref, buy_offset, sell_offset, limit_threshold, tick)
        est_cash = float(price) * int(vol)
        if min_sell_cash > 0 and est_cash < min_sell_cash:
            continue
        orders.append({
            "symbol": sym,
            "side": "SELL",
            "volume": int(vol),
            "price": float(price),
            "type": "limit",
        })

    if allow_buy:
        existing_names: list[tuple[str, int]] = []
        new_names: list[tuple[str, int]] = []
        for t in targets:
            sym = t["symbol"]
            tgt = int(t["target_shares"]) if int(t["target_shares"]) > 0 else 0
            cur = int(current_positions.get(sym, 0))
            if tgt > cur and sym not in set(limit_up_syms):
                vol = ((tgt - cur) // lot) * lot
                if vol <= 0:
                    continue
                if cur > 0:
                    existing_names.append((sym, vol))
                else:
                    new_names.append((sym, vol))
        new_names = new_names[:allowed_new_slots]
        for sym, vol in existing_names + new_names:
            ref = ref_prices.get(sym)
            if ref is None or ref <= 0:
                continue
            if cfg.get("order", {}).get("mode", "auto") == "manual":
                price = compute_manual_price("BUY", ref, buy_offset, sell_offset, tick)
            else:
                if price_source == "tushare":
                    quote = quote_details.get(sym, {})
                    rt_price = float(quote.get("ask") or quote.get("price") or quote.get("bid") or ref)
                    pre_close = float(quote.get("pre_close") or ref)
                    if rt_price > 0 and pre_close > 0:
                        price = compute_limit_price_from_rt_preclose(
                            "BUY",
                            rt_price,
                            pre_close,
                            buy_offset,
                            sell_offset,
                            limit_threshold,
                            tick,
                        )
                    else:
                        price = compute_allowed_price("BUY", ref, buy_offset, sell_offset, limit_threshold, tick)
                else:
                    price = compute_allowed_price("BUY", ref, buy_offset, sell_offset, limit_threshold, tick)
            est_cash = float(price) * int(vol)
            if min_buy_cash > 0 and est_cash < min_buy_cash:
                continue
            orders.append({
                "symbol": sym,
                "side": "BUY",
                "volume": int(vol),
                "price": float(price),
                "type": "limit",
            })

    return orders


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logs_dir = ensure_logs_dir(cfg)

    sig = load_json(args.signals)
    signals = sig.get("signals", [])
    # build rank map by (score, confidence) desc for prioritization
    signals_sorted = sorted(signals, key=lambda x: (x.get("score", 0.0), x.get("confidence", 0.0)), reverse=True)
    rank_map: dict[str, int] = {}
    for idx, s in enumerate(signals_sorted):
        sym = s.get("symbol")
        if isinstance(sym, str):
            rank_map[sym] = idx

    # risk
    allow_buy = True
    limit_up_syms: list[str] = []
    limit_down_syms: list[str] = []
    if args.risk_state and os.path.exists(args.risk_state):
        rs = load_json(args.risk_state)
        allow_buy = bool(rs.get("allow_buy", True))
        limit_up_syms = list(rs.get("limit_up_symbols", []))
        limit_down_syms = list(rs.get("limit_down_symbols", []))

    # capital
    total_capital = None
    if args.capital_override:
        try:
            total_capital = float(args.capital_override)
        except Exception:
            total_capital = None
    if total_capital is None:
        total_capital = float(cfg.get("capital", {}).get("initial", 300000))

    # current positions (symbol -> shares). If not provided, assume empty (first build)
    current_positions: dict[str, int] = {}
    if args.current_positions and os.path.exists(args.current_positions):
        cur = load_json(args.current_positions)
        # expect {"positions":[{"symbol":"SHSE.600000","shares":1000}, ...]}
        for p in cur.get("positions", []):
            sym = p.get("symbol")
            sh = int(p.get("shares", 0))
            if sym and sh > 0:
                current_positions[sym] = sh
    # auto-fetch from gmtrade if still empty (typical for day-2 rebalance)
    if len(current_positions) == 0:
        auto_cur = fetch_current_positions_from_gm(cfg, args.account_id)
        if auto_cur:
            current_positions = auto_cur

    targets, initial_buys, ref_prices_map, quote_details, invest_cap, total_target_value = build_targets(
        signals,
        cfg,
        allow_buy,
        limit_up_syms,
        limit_down_syms,
        total_capital,
    )

    from utils import gm_to_instrument
    qlib_instruments = []
    for t in targets:
        ins = t.get("instrument")
        if isinstance(ins, str):
            qlib_instruments.append(ins)
    for sym in current_positions:
        ins = gm_to_instrument(sym)
        if ins:
            qlib_instruments.append(ins)

    price_source = cfg["order"].get("price_source", "qlib_close")
    if price_source == "tushare":
        gm_needed = set(current_positions.keys())
        gm_needed.update([t["symbol"] for t in targets])
        missing = [gm for gm in gm_needed if gm not in ref_prices_map]
        if missing:
            extra_quotes = fetch_tushare_quotes(missing, src=cfg["order"].get("tushare_src", "sina"))
            for gm_symbol, info in extra_quotes.items():
                price = float(info.get("price") or info.get("ask") or info.get("bid") or info.get("pre_close") or 0.0)
                if price > 0:
                    ref_prices_map[gm_symbol] = price
                if info:
                    quote_details.setdefault(gm_symbol, {}).update(info)
        ref_prices = ref_prices_map
    else:
        ref_prices = read_close_prices_from_h5(cfg["paths"]["daily_pv"], list(set(qlib_instruments)))

    orders = dynamic_adjust_with_positions(
        targets,
        current_positions,
        cfg,
        ref_prices,
        quote_details,
        price_source,
        allow_buy,
        limit_up_syms,
        limit_down_syms,
    )

    remaining_cash = invest_cap - sum(
        float(od["price"]) * int(od.get("volume", 0)) for od in orders if od.get("side") == "BUY"
    )

    # strategy-side budget enforcement for final BUY list
    float(cfg["portfolio"]["invest_ratio"])  # 0.95
    invest_cap = invest_cap
    buy_orders = [od for od in orders if od["side"] == "BUY"]
    sell_orders = [od for od in orders if od["side"] == "SELL"]
    budget = invest_cap
    final_buys: list[dict] = []
    for od in buy_orders:
        est_cost = float(od["price"]) * int(od["volume"])  # conservative
        if est_cost <= budget + 1e-6:
            final_buys.append(od)
            budget -= est_cost
        else:
            continue
    orders = sell_orders + final_buys

    # Optional: raise cash via trimming worst-ranked existing names when no sells present but planned buys exceed available cash
    if allow_buy:
        try:
            from gm_api_wrapper import get_available_cash_amount  # type: ignore
            available_cash = float(get_available_cash_amount())
        except Exception:  # noqa: E722
            available_cash = 0.0
        total_buy_need = 0.0
        for od in final_buys:
            total_buy_need += float(od.get("price", 0.0)) * int(od.get("volume", 0))
        force_raise_cash = bool(cfg.get("rebalance_trigger", {}).get("force_raise_cash_when_shortfall", True))
        n_drop = int(cfg["portfolio"]["n_drop"])  # reuse cap
        lot = int(cfg["portfolio"]["lot"])  # 100
        buy_offset = float(cfg["order"]["buy_limit_offset"])  # 0.02
        sell_offset = float(cfg["order"]["sell_limit_offset"])  # -0.02
        limit_threshold = float(cfg["order"]["limit_threshold"])  # 0.095
        tick = float(cfg["order"]["clamp_tick"])  # 0.01
        order_cfg = cfg.get("order", {})
        min_sell_cash = float(order_cfg.get("min_sell_cash") or order_cfg.get("min_sell_notional") or 0.0)
        if force_raise_cash and len(sell_orders) == 0 and total_buy_need > available_cash + 1e-6:
            need = total_buy_need - available_cash
            # candidates: current held names that are in targets (remain) and not limit-down
            target_syms = {t["symbol"] for t in targets}
            candidates = [sym for sym in current_positions if sym in target_syms and sym not in set(limit_down_syms)]
            # sort by worse rank (larger idx)
            candidates.sort(key=lambda s: rank_map.get(s, 10**9), reverse=True)
            extra_sells: list[dict] = []
            for sym in candidates:
                if need <= 1e-6 or len(extra_sells) >= n_drop:
                    break
                price_ref = ref_prices.get(sym)
                if price_ref is None or price_ref <= 0:
                    continue
                cur_sh = int(current_positions.get(sym, 0))
                max_lot_vol = (cur_sh // lot) * lot
                if max_lot_vol <= 0:
                    continue
                # sell just enough lots to cover remaining need
                lots_need = int(math.ceil(need / max(price_ref * lot, 1e-6)))
                lots_need = max(1, lots_need)
                vol = min(max_lot_vol, lots_need * lot)
                px = compute_allowed_price("SELL", price_ref, buy_offset, sell_offset, limit_threshold, tick)
                est_cash = float(px) * int(vol)
                if min_sell_cash > 0 and est_cash < min_sell_cash:
                    continue
                extra_sells.append({
                    "symbol": sym,
                    "side": "SELL",
                    "volume": int(vol),
                    "price": float(px),
                    "type": "limit",
                })
                need -= vol * price_ref
            if extra_sells:
                orders = extra_sells + orders

    date_str = sig.get("date") or "AUTO"
    targets_out = args.targets or os.path.join(logs_dir, f"targets_{date_str}.json")
    orders_out = args.orders or os.path.join(logs_dir, f"orders_{date_str}.json")

    with open(targets_out, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "targets": targets, "invest_capital": invest_cap, "total_capital": total_target_value, "remaining_cash": remaining_cash}, f, ensure_ascii=False, indent=2)

    with open(orders_out, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "orders": orders}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"status": "ok", "targets": len(targets), "orders": len(orders), "targets_out": targets_out, "orders_out": orders_out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
