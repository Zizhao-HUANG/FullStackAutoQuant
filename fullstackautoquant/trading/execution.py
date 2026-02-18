import argparse
import json
import os
from datetime import datetime, time

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # noqa: E722
    ZoneInfo = None  # type: ignore
from pathlib import Path

from dotenv import load_dotenv
from utils import (
    clamp_volume_to_lot,
    compute_auction_price,
    compute_manual_price,
    compute_open_mid_price,
    ensure_logs_dir,
    gm_to_ts_code,
    load_config,
    max_order_volume_for_symbol,
    min_order_lot_for_symbol,
    save_json,
)

# trade api
try:
    from gmtrade.api import (
        OrderSide_Buy,
        OrderSide_Sell,
        OrderType_Limit,
        PositionEffect_Close,
        PositionEffect_Open,
        account,
        get_cash,
        login,
        order_volume,
        set_endpoint,
        set_token,
    )
except Exception:  # noqa: E722
    set_token = None
    set_endpoint = None
    account = None
    login = None
    order_volume = None
    OrderSide_Buy = None
    OrderSide_Sell = None
    OrderType_Limit = None
    PositionEffect_Open = None
    PositionEffect_Close = None
    get_cash = None

# tushare realtime
try:
    import tushare as ts
except Exception:  # noqa: E722
    ts = None


def parse_args():
    p = argparse.ArgumentParser(description="GM API wrapper with Tushare realtime pricing")
    p.add_argument("--orders", required=True, help="orders JSON path")
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--place", action="store_true", help="actually place orders")
    p.add_argument("--alias", default="", help="optional account alias")
    p.add_argument("--src", default="sina", choices=["sina", "dc"], help="tushare source (sina primary with dc fallback)")
    # execution behavior extensions
    p.add_argument("--auction-mode", action="store_true", help="use opening auction pricing when within 09:15–09:25 Beijing time")
    p.add_argument("--max_slices_open", type=int, default=1, help="max slices for open execution after 09:30 (default 1; set 2 when condition triggers)")
    p.add_argument("--open_eps", type=float, default=0.0025, help="epsilon tilt for midpoint pricing in continuous session (e.g., 0.0025 for 0.25%)")
    # explicit GM overrides
    p.add_argument("--account-id", default=None, help="override GM account id (env GM_ACCOUNT_ID otherwise)")
    return p.parse_args()


def load_orders(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gm_login(cfg: dict, alias: str = "", account_id_override: str = None):
    # Load .env from multiple likely locations to reduce mis-config risk
    load_dotenv()  # current working directory
    here = Path(__file__).resolve().parent
    for p in [here / ".env", here.parent / ".env", here.parent.parent / ".env"]:
        if p.exists():
            load_dotenv(p.as_posix(), override=True)
    endpoint = os.getenv("GM_ENDPOINT") or cfg.get("gm", {}).get("endpoint")
    token = os.getenv("GM_TOKEN") or cfg.get("gm", {}).get("token")
    acc_id = account_id_override or os.getenv("GM_ACCOUNT_ID") or cfg.get("gm", {}).get("account_id")
    if not all([endpoint, token, acc_id]):
        raise RuntimeError("Missing GM credentials: endpoint/token/account_id")
    if set_endpoint is None:
        raise RuntimeError("gmtrade is not installed; pip install gmtrade")
    set_endpoint(endpoint)
    set_token(token)
    a1 = account(account_id=acc_id, account_alias=alias)
    login(a1)
    return acc_id


def fetch_realtime_quotes_tushare(ts_codes: list[str], src: str = "sina") -> dict[str, dict[str, float]]:
    if ts is None:
        raise RuntimeError("tushare not installed; pip install tushare>=1.3.3")
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    ts_token = os.getenv("TUSHARE") or os.getenv("TS_TOKEN")
    if not ts_token:
        raise RuntimeError("Missing Tushare token in env var TUSHARE or TS_TOKEN")
    ts.set_token(ts_token)

    out: dict[str, dict[str, float]] = {}
    batch_size = 1 if src == "dc" else 50
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i : i + batch_size]
        query_arg = ",".join(batch) if src == "sina" else batch[0]
        df = ts.realtime_quote(ts_code=query_arg, src=src)
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            rec = row.to_dict()
            ts_code_val = rec.get("ts_code") or rec.get("TS_CODE")
            if not ts_code_val:
                continue
            # robust field extraction across sources
            def _get_first(keys, default=0.0):
                for k in keys:
                    if k in rec and rec.get(k) not in (None, "", " "):
                        return rec.get(k)
                return default
            price = _get_first(["price", "PRICE", "last", "LAST"]) or 0.0
            pre_close = _get_first(["pre_close", "PRE_CLOSE", "preclose", "PRECLOSE", "yesterday_close", "YESTERDAY_CLOSE"]) or 0.0
            bid = _get_first(["bid", "BID", "B1_P", "b1_p", "buy", "BUY"]) or 0.0
            ask = _get_first(["ask", "ASK", "A1_P", "a1_p", "sell", "SELL"]) or 0.0
            limit_up = _get_first(["limit_up", "LIMIT_UP", "up_limit", "UP_LIMIT", "high_limit", "HIGH_LIMIT"]) or 0.0
            limit_down = _get_first(["limit_down", "LIMIT_DOWN", "down_limit", "DOWN_LIMIT", "low_limit", "LOW_LIMIT"]) or 0.0
            out[str(ts_code_val)] = {
                "price": float(price) if price else 0.0,
                "pre_close": float(pre_close) if pre_close else 0.0,
                "bid": float(bid) if bid else 0.0,
                "ask": float(ask) if ask else 0.0,
                "limit_up": float(limit_up) if limit_up else 0.0,
                "limit_down": float(limit_down) if limit_down else 0.0,
            }
    return out


def fetch_realtime_with_fallback(ts_codes: list[str]) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Try sina first, then fallback dc for missing or invalid records. Return (quotes, used_fallback_codes)."""
    used_fallback: set[str] = set()
    quotes: dict[str, dict[str, float]] = {}
    # try sina
    try:
        quotes = fetch_realtime_quotes_tushare(ts_codes, src="sina")
    except Exception:
        quotes = {}
    # detect missing or invalid
    need_dc: list[str] = []
    for code in ts_codes:
        rec = quotes.get(code)
        if rec is None:
            need_dc.append(code)
            continue
        pre_close = float(rec.get("pre_close", 0.0))
        price = float(rec.get("price", 0.0))
        bid = float(rec.get("bid", 0.0))
        ask = float(rec.get("ask", 0.0))
        limu = float(rec.get("limit_up", 0.0))
        limd = float(rec.get("limit_down", 0.0))
        if pre_close <= 0 or price <= 0 and bid <= 0 and ask <= 0:
            need_dc.append(code)
        elif limu <= 0 or limd <= 0:
            # try dc to fill hard bounds even if price is available
            need_dc.append(code)
    # fetch dc one-by-one
    for code in need_dc:
        try:
            dc_q = fetch_realtime_quotes_tushare([code], src="dc")
            if dc_q:
                used_fallback.add(code)
                if code not in quotes:
                    quotes[code] = dc_q.get(code, {})
                else:
                    rec = quotes[code]
                    fb = dc_q.get(code, {})
                    for k in ["price", "pre_close", "bid", "ask", "limit_up", "limit_down"]:
                        if float(rec.get(k, 0.0)) <= 0 and float(fb.get(k, 0.0)) > 0:
                            rec[k] = float(fb.get(k))
        except Exception:
            continue
    return quotes, used_fallback


def get_available_cash_amount() -> float:
    if get_cash is None:
        return 0.0
    try:
        cash = get_cash()
        if hasattr(cash, "available"):
            return float(cash.available)
        if isinstance(cash, dict) and "available" in cash:
            return float(cash["available"])  # type: ignore
    except Exception:
        return 0.0
    return 0.0


def _now_in_beijing() -> datetime:
    if ZoneInfo is None:
        # fallback to system time (assumed already in Beijing on servers)
        return datetime.now()
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def _is_in_auction_window(now_sh: datetime) -> bool:
    t1 = time(9, 15, 0)
    t2 = time(9, 25, 0)
    return t1 <= now_sh.time() <= t2


def _is_before_auction_cut(now_sh: datetime) -> bool:
    # soft cut to avoid 09:20–09:25 no-cancel; users should submit by ~09:19:30
    t1 = time(9, 19, 30)
    return now_sh.time() <= t1


def place_orders(orders: list[dict], do_place: bool, cfg: dict, src: str, auction_mode: bool, max_slices_open: int, open_eps: float) -> list[dict]:
    receipts: list[dict] = []
    # split sells and buys to avoid cash freeze issue
    sells = [od for od in orders if str(od.get("side", "")).upper() == "SELL"]
    buys = [od for od in orders if str(od.get("side", "")).upper() == "BUY"]

    gm_symbols = [od["symbol"] for od in orders]
    ts_codes = [gm_to_ts_code(s) for s in gm_symbols if gm_to_ts_code(s)]
    if src == "dc":
        quotes = fetch_realtime_quotes_tushare(ts_codes, src="dc") if ts_codes else {}
        used_fallback: set[str] = set()
    else:
        quotes, used_fallback = fetch_realtime_with_fallback(ts_codes) if ts_codes else ({}, set())

    buy_offset = float(cfg["order"]["buy_limit_offset"])
    sell_offset = float(cfg["order"]["sell_limit_offset"])
    limit_threshold = float(cfg["order"]["limit_threshold"])  # 0.095
    tick = float(cfg["order"]["clamp_tick"])  # 0.01

    mode_manual = cfg.get("order", {}).get("mode", "auto") == "manual"

    fee_rate_open = float(cfg.get("order", {}).get("open_cost", 0.0005))
    min_cost = float(cfg.get("order", {}).get("min_cost", 5))

    # per-name cap (based on invest_cap*max_w)
    invest_ratio = float(cfg["portfolio"]["invest_ratio"])  # 0.95
    total_capital = float(cfg.get("capital", {}).get("initial", 300000))
    invest_cap = total_capital * invest_ratio
    max_w = float(cfg["portfolio"]["max_weight"])  # 0.05
    lot = int(cfg["portfolio"]["lot"])  # 100
    per_name_cap_amt = invest_cap * max_w

    manual_budget = invest_cap if mode_manual else None

    def compute_price_note(sym: str, side: str, fallback_used: bool, supplied_price: float) -> tuple[float, str]:
        ts_code = gm_to_ts_code(sym)
        q = quotes.get(ts_code, {}) if ts_code else {}
        pre_close = float(q.get("pre_close", 0.0))
        bid = float(q.get("bid", 0.0))
        ask = float(q.get("ask", 0.0))
        limit_up_rt = float(q.get("limit_up", 0.0))
        limit_down_rt = float(q.get("limit_down", 0.0))
        rt_price = float(q.get("price", 0.0))

        if mode_manual:
            ref_price = pre_close if pre_close > 0 else supplied_price if supplied_price and supplied_price > 0 else rt_price if rt_price > 0 else tick
            if ref_price <= 0:
                ref_price = tick
            price_manual = compute_manual_price(side, ref_price, buy_offset, sell_offset, tick)
            note_manual_parts = ["MODE_MANUAL"]
            if pre_close <= 0 and not (supplied_price and supplied_price > 0):
                note_manual_parts.append("NO_RT_QUOTE")
            if fallback_used:
                note_manual_parts.append("FALLBACK_DC")
            return price_manual, "|".join(note_manual_parts)

        if pre_close <= 0:
            note_auto = ["NO_RT_QUOTE"]
            if fallback_used:
                note_auto.append("FALLBACK_DC")
            return supplied_price if supplied_price > 0 else tick, "|".join(note_auto)

        now_sh = _now_in_beijing()
        in_auction = auction_mode and _is_in_auction_window(now_sh)
        if in_auction and _is_before_auction_cut(now_sh):
            price = compute_auction_price(
                side,
                sym,
                pre_close=pre_close,
                tick=tick,
                limit_up_override=limit_up_rt if limit_up_rt > 0 else None,
                limit_down_override=limit_down_rt if limit_down_rt > 0 else None,
            )
            note_mode = "MODE_AUCTION"
        else:
            price = compute_open_mid_price(
                side,
                sym,
                bid=bid,
                ask=ask,
                rt_price=rt_price,
                pre_close=pre_close,
                eps=open_eps,
                tick=tick,
                limit_up_override=limit_up_rt if limit_up_rt > 0 else None,
                limit_down_override=limit_down_rt if limit_down_rt > 0 else None,
            )
            note_mode = "MODE_OPEN"
        if fallback_used:
            note_mode = f"{note_mode}|FALLBACK_DC"
        return price, note_mode

    def _extract_order_id(odr) -> str:
        for key in [
            "cl_ord_id",
            "order_id",
            "orderno",
            "id",
            "client_id",
            "clOrderId",
            "clOrderID",
        ]:
            try:
                if hasattr(odr, key):
                    val = getattr(odr, key)
                    if isinstance(val, (str, int)) and str(val):
                        return str(val)
            except Exception:  # noqa: E722
                pass
        try:
            # try dict-like
            if isinstance(odr, dict):
                for key in ["cl_ord_id", "order_id", "orderno", "id", "client_id"]:
                    if key in odr and odr[key]:
                        return str(odr[key])
        except Exception:  # noqa: E722
            pass
        return ""

    # helper to split by lot for slicing
    def split_by_slices(sym: str, volume: int, slices: int) -> list[int]:
        base = max(1, int(slices))
        vol = clamp_volume_to_lot(sym, volume)
        if vol <= 0 or base <= 1:
            return [vol]
        parts: list[int] = []
        lot = min_order_lot_for_symbol(sym)
        per = max(lot, (vol // base) // lot * lot)
        allocated = 0
        for _i in range(base - 1):
            parts.append(per)
            allocated += per
        parts.append(max(0, vol - allocated))
        # ensure each part aligns to lot
        parts = [((p // lot) * lot) for p in parts]
        parts = [p for p in parts if p > 0]
        if not parts:
            return [0]
        if sum(parts) != vol:
            # adjust last part to match total (still multiple of lot)
            diff = vol - sum(parts)
            parts[-1] = max(0, parts[-1] + diff)
        return parts

    # 1) place SELL first
    for od in sells:
        sym = od["symbol"]
        side = "SELL"
        vol = int(od["volume"]) if str(od["volume"]).isdigit() else int(float(od["volume"]))
        # auction uses single shot; open session can slice
        now_sh = _now_in_beijing()
        slices = 1 if (auction_mode and _is_in_auction_window(now_sh) and _is_before_auction_cut(now_sh)) else max(1, int(max_slices_open))
        vols = split_by_slices(sym, vol, slices)
        fb = gm_to_ts_code(sym) in used_fallback
        for vol_i in vols:
            if vol_i <= 0:
                continue
            price, note = compute_price_note(sym, side, fb, float(od.get("price", 0.0)))
            if not do_place:
                receipts.append({"symbol": sym, "side": side, "volume": vol_i, "price": price, "status": "DRY_RUN", "note": note})
                continue
            try:
                side_enum = OrderSide_Sell
                effect_enum = PositionEffect_Close
                odr = order_volume(symbol=sym, volume=vol_i, side=side_enum, order_type=OrderType_Limit, position_effect=effect_enum, price=price)
                oid = _extract_order_id(odr)
                if not oid:
                    # Do not re-submit to avoid duplicate orders; log unknown ID
                    receipts.append({"symbol": sym, "side": side, "volume": vol_i, "price": price, "status": "SUBMITTED", "order_id": "", "note": (note or "") + "|NO_ORDER_ID"})
                else:
                    receipts.append({"symbol": sym, "side": side, "volume": vol_i, "price": price, "status": "SUBMITTED", "order_id": oid, "note": note})
            except Exception as e1:  # noqa: E722
                try:
                    odr = order_volume(symbol=sym, volume=-vol_i, side=side_enum, order_type=OrderType_Limit, position_effect=effect_enum, price=price)
                    oid = _extract_order_id(odr)
                    note2 = f"FALLBACK:{note}" if note else "FALLBACK"
                    if oid:
                        receipts.append({"symbol": sym, "side": side, "volume": vol_i, "price": price, "status": "SUBMITTED", "order_id": oid, "note": note2})
                    else:
                        receipts.append({"symbol": sym, "side": side, "volume": vol_i, "price": price, "status": "SUBMITTED", "order_id": "", "note": note2 + "|NO_ORDER_ID"})
                except Exception as e2:  # noqa: E722
                    receipts.append({"symbol": sym, "side": side, "volume": vol_i, "price": price, "status": "ERROR", "error": f"{e1} | {e2}", "note": note})

    available_cash = get_available_cash_amount() if do_place else 1e18
    cash_check_valid = True
    if not do_place:
        cash_check_valid = False
        available_cash = 1e18
    elif mode_manual:
        cash_check_valid = False
        available_cash = manual_budget if manual_budget is not None else invest_cap
    elif available_cash is None or available_cash <= 0:
        cash_check_valid = False
        available_cash = 1e18

    # 2) BUY within both per-name cap and available cash
    for od in buys:
        sym = od["symbol"]
        side = "BUY"
        vol = int(od["volume"]) if str(od["volume"]).isdigit() else int(float(od["volume"]))
        fb = gm_to_ts_code(sym) in used_fallback
        now_sh = _now_in_beijing()
        slices = 1 if (auction_mode and _is_in_auction_window(now_sh) and _is_before_auction_cut(now_sh)) else max(1, int(max_slices_open))
        price, note = compute_price_note(sym, side, fb, float(od.get("price", 0.0)))
        # align to market lot and enforce max volume
        min_lot = min_order_lot_for_symbol(sym)
        max_vol = max_order_volume_for_symbol(sym)
        vol = clamp_volume_to_lot(sym, vol)
        if vol < min_lot:
            receipts.append({"symbol": sym, "side": side, "volume": vol, "price": price, "status": "SKIPPED_MIN_LOT", "note": note})
            continue
        if vol > max_vol:
            vol = (max_vol // min_lot) * min_lot
        # cap per name by amount
        max_vol_by_cap = int((per_name_cap_amt // (price * min_lot)) * min_lot) if price > 0 else 0
        if mode_manual and price > 0 and max_vol_by_cap < min_lot:
            # Manual mode: if the amount cap cannot reach the minimum lot, force minimum lot size
            max_vol_by_cap = min_lot
        if max_vol_by_cap <= 0:
            receipts.append({"symbol": sym, "side": side, "volume": 0, "price": price, "status": "SKIPPED_CAP_ZERO", "note": note})
            continue
        if vol > max_vol_by_cap:
            vol = max_vol_by_cap
        est_cost = price * vol
        est_fee = max(min_cost, est_cost * fee_rate_open)
        total_need = est_cost + est_fee
        vols = split_by_slices(sym, vol, slices)
        if do_place:
            total_need_all = 0.0
            per_costs: list[tuple[int, float]] = []
            for v in vols:
                c = price * v
                f = max(min_cost, c * fee_rate_open)
                per_costs.append((v, c + f))
                total_need_all += c + f
            available_limit = available_cash if cash_check_valid else 1e18
            if manual_budget is not None:
                available_limit = min(available_limit, manual_budget)
            if total_need_all > available_limit + 1e-6:
                receipts.append({"symbol": sym, "side": side, "volume": sum(vols), "price": price, "status": "SKIPPED_INSUFFICIENT_CASH", "need": round(total_need_all, 2), "available": round(available_cash, 2), "note": note + "|NO_CASH_INFO" if not cash_check_valid else note})
                continue
            for v, need_v in per_costs:
                if v <= 0:
                    continue
                try:
                    side_enum = OrderSide_Buy
                    effect_enum = PositionEffect_Open
                    odr = order_volume(symbol=sym, volume=v, side=side_enum, order_type=OrderType_Limit, position_effect=effect_enum, price=price)
                    oid = _extract_order_id(odr)
                    if not oid:
                        # Do not re-submit to avoid duplicates
                        receipts.append({"symbol": sym, "side": side, "volume": v, "price": price, "status": "SUBMITTED", "order_id": "", "note": (note if cash_check_valid else (note + "|NO_CASH_CHECK")) + "|NO_ORDER_ID"})
                        if cash_check_valid:
                            available_cash -= need_v
                        if manual_budget is not None:
                            manual_budget -= need_v
                    else:
                        receipts.append({"symbol": sym, "side": side, "volume": v, "price": price, "status": "SUBMITTED", "order_id": oid, "note": note if cash_check_valid else (note + "|NO_CASH_CHECK")})
                        if cash_check_valid:
                            available_cash -= need_v
                        if manual_budget is not None:
                            manual_budget -= need_v
                except Exception as e1:  # noqa: E722
                    try:
                        odr = order_volume(symbol=sym, volume=v, side=side_enum, order_type=OrderType_Limit, position_effect=effect_enum, price=price)
                        oid = _extract_order_id(odr)
                        note2 = (f"FALLBACK:{note}") if cash_check_valid else (f"FALLBACK:{note}|NO_CASH_CHECK")
                        if oid:
                            receipts.append({"symbol": sym, "side": side, "volume": v, "price": price, "status": "SUBMITTED", "order_id": oid, "note": note2})
                            if cash_check_valid:
                                available_cash -= need_v
                            if manual_budget is not None:
                                manual_budget -= need_v
                        else:
                            receipts.append({"symbol": sym, "side": side, "volume": v, "price": price, "status": "SUBMITTED", "order_id": "", "note": note2 + "|NO_ORDER_ID"})
                    except Exception as e2:  # noqa: E722
                        receipts.append({"symbol": sym, "side": side, "volume": v, "price": price, "status": "ERROR", "error": f"{e1} | {e2}", "note": note})
        else:
            for v in vols:
                if v <= 0:
                    continue
                receipts.append({"symbol": sym, "side": side, "volume": v, "price": price, "status": "DRY_RUN", "note": note})

    return receipts


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logs_dir = ensure_logs_dir(cfg)

    od = load_orders(args.orders)
    date_str = od.get("date") or "AUTO"
    orders = od.get("orders", [])

    placed = False
    acc_id = None
    if args.place and len(orders) > 0:
        acc_id = gm_login(cfg, alias=args.alias, account_id_override=(args.account_id or None))
        placed = True

    receipts = place_orders(orders, do_place=placed, cfg=cfg, src=args.src, auction_mode=bool(args.auction_mode), max_slices_open=int(args.max_slices_open), open_eps=float(args.open_eps))
    out_path = os.path.join(logs_dir, f"receipts_{date_str}.json")
    save_json({"date": date_str, "receipts": receipts, "placed": placed, "account_id": acc_id}, out_path)
    print(json.dumps({"status": "ok", "out": out_path, "placed": placed, "num_orders": len(orders)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
