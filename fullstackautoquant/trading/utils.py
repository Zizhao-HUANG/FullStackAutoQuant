import json
import math
import os

import pandas as pd
import yaml


def load_config(config_path: str | None) -> dict:
    """Load YAML config; if None, use default path next to this file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    elif not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(os.getcwd(), config_path))
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # ensure logs dir
    logs_dir = cfg.get("paths", {}).get("logs_dir")
    if logs_dir:
        try:
            os.makedirs(logs_dir, exist_ok=True)
        except OSError:
            pass
    price_source = cfg.get("order", {}).get("price_source")
    if isinstance(price_source, str) and price_source.lower() == "tushare":
        token = cfg.get("paths", {}).get("tushare_token") or os.getenv("TUSHARE")
        if token:
            try:
                import tushare as ts  # type: ignore

                ts.set_token(str(token))
            except Exception:  # pragma: no cover
                pass
    return cfg


def fetch_tushare_quotes(gm_symbols: list[str], src: str = "sina") -> dict[str, dict[str, float]]:
    """Fetch realtime quotes from Tushare and return mapping of GM symbols."""

    gm_symbols = [s for s in gm_symbols if isinstance(s, str)]
    if not gm_symbols:
        return {}

    ts_codes = []
    gm_map: dict[str, str] = {}
    for gm_symbol in gm_symbols:
        ts_code = gm_to_ts_code(gm_symbol)
        if ts_code:
            ts_codes.append(ts_code)
            gm_map[ts_code.upper()] = gm_symbol

    if not ts_codes:
        return {}

    try:
        import tushare as ts  # type: ignore
    except Exception:  # pragma: no cover
        return {}

    try:
        df = ts.realtime_quote(ts_code=",".join(ts_codes), src=src)  # type: ignore
    except Exception:  # pragma: no cover
        return {}

    if df is None or df.empty:
        return {}

    result: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        row_dict = {str(col).lower(): row[col] for col in df.columns}
        ts_code = str(row_dict.get("ts_code") or row_dict.get("symbol") or "").upper()
        gm_symbol = gm_map.get(ts_code)
        if not gm_symbol:
            continue
        def _to_float(key: str) -> float:
            try:
                return float(row_dict.get(key, 0.0) or 0.0)
            except Exception:
                return 0.0

        result[gm_symbol] = {
            "price": _to_float("price"),
            "bid": _to_float("bid"),
            "ask": _to_float("ask"),
            "pre_close": _to_float("pre_close"),
            "open": _to_float("open"),
        }
    return result


def instrument_to_gm(symbol: str) -> str | None:
    symbol = symbol.strip().upper()
    if len(symbol) != 8:
        return None
    market = symbol[:2]
    code = symbol[2:]
    if not code.isdigit():
        return None
    if market == "SH":
        return f"SHSE.{code}"
    if market == "SZ":
        return f"SZSE.{code}"
    return None


def gm_to_instrument(gm_symbol: str) -> str | None:
    gm_symbol = gm_symbol.strip().upper()
    if gm_symbol.startswith("SHSE."):
        return f"SH{gm_symbol.split('.')[1]}"
    if gm_symbol.startswith("SZSE."):
        return f"SZ{gm_symbol.split('.')[1]}"
    return None


def gm_to_ts_code(gm_symbol: str) -> str | None:
    """Map GM symbol to Tushare ts_code, e.g., SHSE.600000 -> 600000.SH"""
    gm_symbol = gm_symbol.strip().upper()
    if gm_symbol.startswith("SHSE."):
        return f"{gm_symbol.split('.')[1]}.SH"
    if gm_symbol.startswith("SZSE."):
        return f"{gm_symbol.split('.')[1]}.SZ"
    return None


def ts_code_to_gm(ts_code: str) -> str | None:
    ts_code = ts_code.strip().upper()
    if ts_code.endswith(".SH"):
        return f"SHSE.{ts_code.split('.')[0]}"
    if ts_code.endswith(".SZ"):
        return f"SZSE.{ts_code.split('.')[0]}"
    return None


def round_price(price: float, tick: float, mode: str) -> float:
    if tick <= 0:
        return round(price, 2)
    if mode == "up":
        return math.ceil(price / tick) * tick
    if mode == "down":
        return math.floor(price / tick) * tick
    return round(round(price / tick) * tick, 2)


def compute_allowed_price(side: str, ref_price: float, buy_offset: float, sell_offset: float, limit_threshold: float, tick: float) -> float:
    ref_price = float(ref_price)
    if side == "BUY":
        buy_offset = abs(buy_offset)
        raw = ref_price * (1.0 + buy_offset)
        exch_upper = max(ref_price * 1.02, ref_price + 10.0 * tick)
        limit_upper = ref_price * (1.0 + limit_threshold)
        price = min(raw, exch_upper, limit_upper)
        return round_price(price, tick, mode="up")
    else:
        sell_offset = abs(sell_offset)
        raw = ref_price * (1.0 - sell_offset)
        exch_lower = min(ref_price * 0.98, ref_price - 10.0 * tick)
        limit_lower = ref_price * (1.0 - limit_threshold)
        price = max(raw, exch_lower, limit_lower)
        return round_price(price, tick, mode="down")


def compute_limit_price_from_rt_preclose(side: str, rt_price: float, pre_close: float, buy_offset: float, sell_offset: float, limit_threshold: float, tick: float) -> float:
    """Compute limit price using realtime price for offset and pre-close for exchange bounds/clamp."""
    rt_price = float(rt_price)
    pre_close = float(pre_close)
    if side == "BUY":
        buy_offset = abs(buy_offset)
        raw = rt_price * (1.0 + buy_offset)
        exch_upper = max(pre_close * 1.02, pre_close + 10.0 * tick)
        limit_upper = pre_close * (1.0 + limit_threshold)
        price = min(raw, exch_upper, limit_upper)
        return round_price(price, tick, mode="up")
    else:
        sell_offset = abs(sell_offset)
        raw = rt_price * (1.0 - sell_offset)
        exch_lower = min(pre_close * 0.98, pre_close - 10.0 * tick)
        limit_lower = pre_close * (1.0 - limit_threshold)
        price = max(raw, exch_lower, limit_lower)
        return round_price(price, tick, mode="down")


def board_limit_ratio_for_symbol(gm_symbol: str) -> float:
    """Return daily up/down limit ratio by board.
    - STAR (SSE 688/689) and ChiNext (SZSE 300/301) use 20%
    - Default A shares use 10%
    """
    s = gm_symbol.upper()
    if s.startswith("SHSE.688") or s.startswith("SHSE.689"):
        return 0.20
    if s.startswith("SZSE.300") or s.startswith("SZSE.301"):
        return 0.20
    return 0.10


def compute_auction_price(
    side: str,
    gm_symbol: str,
    pre_close: float,
    tick: float,
    gamma_10pct: float = 0.097,
    gamma_20pct: float = 0.197,
    limit_up_override: float = None,
    limit_down_override: float = None,
) -> float:
    """Market-like limit price for opening auction (09:15–09:25).
    - For 10% boards use gamma=0.097; for 20% boards use gamma=0.197
    - Buy: min(limit_up - tick, round_to_tick(pre_close*(1+gamma)))
    - Sell: max(limit_down + tick, round_to_tick(pre_close*(1-gamma)))
    """
    ex_ratio = board_limit_ratio_for_symbol(gm_symbol)
    gamma = gamma_20pct if ex_ratio >= 0.20 else gamma_10pct
    pre_close = float(pre_close)
    if tick <= 0:
        tick = 0.01
    limit_up = float(limit_up_override) if limit_up_override and limit_up_override > 0 else (round_price(pre_close * (1.0 + ex_ratio), tick, mode="nearest") if pre_close > 0 else 0.0)
    limit_down = float(limit_down_override) if limit_down_override and limit_down_override > 0 else (round_price(pre_close * (1.0 - ex_ratio), tick, mode="nearest") if pre_close > 0 else 0.0)
    if side.upper() == "BUY":
        if pre_close > 0:
            target = round_price(pre_close * (1.0 + gamma), tick, mode="nearest")
        else:
            # fallback: without pre_close, lean to safe bound below limit up
            target = limit_up - tick if limit_up > 0 else 0.0
        upper_cap = limit_up - tick if limit_up > 0 else target
        price = min(max(upper_cap, tick), target)
        return round(max(price, tick), 2)
    else:
        if pre_close > 0:
            target = round_price(pre_close * (1.0 - gamma), tick, mode="nearest")
        else:
            target = limit_down + tick if limit_down > 0 else 0.0
        lower_cap = limit_down + tick if limit_down > 0 else target
        price = max(min(lower_cap if pre_close > 0 else target, pre_close if pre_close > 0 else lower_cap), target)
        return round(max(price, tick), 2)


def compute_open_mid_price(
    side: str,
    gm_symbol: str,
    bid: float,
    ask: float,
    rt_price: float,
    pre_close: float,
    eps: float,
    tick: float,
    limit_up_override: float = None,
    limit_down_override: float = None,
) -> float:
    """Continuous auction (09:30+) one-shot pricing using midpoint with small tilt eps.
    - mid = (bid+ask)/2 if both>0 else rt_price if>0 else pre_close
    - Buy: mid*(1+eps), clamped to (limit_up - tick)
    - Sell: mid*(1-eps), clamped to (limit_down + tick)
    """
    ex_ratio = board_limit_ratio_for_symbol(gm_symbol)
    limit_up = float(limit_up_override) if limit_up_override and limit_up_override > 0 else round_price(pre_close * (1.0 + ex_ratio), tick, mode="nearest")
    limit_down = float(limit_down_override) if limit_down_override and limit_down_override > 0 else round_price(pre_close * (1.0 - ex_ratio), tick, mode="nearest")
    mid = None
    if float(bid) > 0 and float(ask) > 0 and float(ask) >= float(bid):
        mid = 0.5 * (float(bid) + float(ask))
    elif float(rt_price) > 0:
        mid = float(rt_price)
    else:
        mid = float(pre_close)
    if side.upper() == "BUY":
        raw = mid * (1.0 + max(0.0, float(eps)))
        raw = round_price(raw, tick, mode="nearest")
        cap = limit_up - tick if limit_up > 0 else raw
        price = min(raw, cap)
        return round(max(price, tick), 2)
    else:
        raw = mid * (1.0 - max(0.0, float(eps)))
        raw = round_price(raw, tick, mode="nearest")
        floor = limit_down + tick if limit_down > 0 else raw
        price = max(raw, floor)
        if pre_close > 0:
            price = max(min(price, pre_close), floor)
        return round(max(price, tick), 2)


def min_order_lot_for_symbol(gm_symbol: str) -> int:
    """Return minimal order quantity per order for limit orders.
    - STAR market (SSE 688/689): 200 shares
    - Default A shares: 100 shares
    """
    s = gm_symbol.upper()
    if s.startswith("SHSE.688") or s.startswith("SHSE.689"):
        return 200
    return 100


def max_order_volume_for_symbol(gm_symbol: str) -> int:
    """Return maximal order quantity per order.
    - STAR market: 100000
    - Default: a large safe cap
    """
    s = gm_symbol.upper()
    if s.startswith("SHSE.688") or s.startswith("SHSE.689"):
        return 100000
    return 1000000


def clamp_volume_to_lot(gm_symbol: str, volume: int) -> int:
    lot = min_order_lot_for_symbol(gm_symbol)
    if volume <= 0:
        return 0
    return (int(volume) // lot) * lot


def read_close_prices_from_h5(h5_path: str, instruments: list[str]) -> dict[str, float]:
    if not os.path.exists(h5_path):
        return {}
    df = pd.read_hdf(h5_path, key="data")
    last_dt = df.index.get_level_values("datetime").max()
    df_last = df.xs(last_dt, level="datetime")
    res: dict[str, float] = {}
    for ins in instruments:
        if ins in df_last.index and "$close" in df_last.columns:
            close_val = float(df_last.loc[ins, "$close"])  # type: ignore
            gm_code = instrument_to_gm(ins)
            if gm_code is not None:
                res[gm_code] = close_val
    return res


def ensure_logs_dir(cfg: dict) -> str:
    logs_dir = cfg.get("paths", {}).get("logs_dir", os.path.join(os.path.dirname(__file__), "logs"))
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except OSError:
        pass
    return logs_dir


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_manual_price(side: str, ref_price: float, buy_offset: float, sell_offset: float, tick: float) -> float:
    """Compute manual trading price using supplied limit price (no extra offset).

    ref_price is the limit price from the strategy layer, align to tick.
    """
    ref = float(ref_price)
    if ref <= 0:
        return 0.0
    if tick <= 0:
        tick = 0.01
    if side.upper() == "BUY":
        return max(round_price(ref, tick, mode="up"), tick)
    else:
        price = round_price(ref, tick, mode="down")
        if price <= 0:
            price = tick
        return price
