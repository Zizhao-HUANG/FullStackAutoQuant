import os
import sys
import argparse
import subprocess
import time as time_mod
import csv
from datetime import datetime, timedelta, time
from typing import List, Tuple, Optional, Set

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # noqa: E722
    ZoneInfo = None  # type: ignore

from utils import load_config, ensure_logs_dir


def parse_args():
    p = argparse.ArgumentParser(description="Daily scheduler (Beijing time) for run_trading_once")
    p.add_argument("--csv", required=True, help="ranked_scores CSV path")
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--place", action="store_true", help="actually place orders")
    p.add_argument("--override_buy", action="store_true", help="override buy block for this run")
    p.add_argument("--src", default="sina", choices=["sina", "dc"], help="tushare source")
    p.add_argument(
        "--times",
        default="09:16:00,09:30:05,09:31:00,13:00:10,14:00:10",
        help="comma-separated Beijing times HH:MM[:SS] to trigger runs (default matches T+1 plan)",
    )
    p.add_argument(
        "--second_slice_times",
        default="09:31:00",
        help="comma-separated Beijing times treated as second-slice (use --max_slices_open=2). Others use 1",
    )
    p.add_argument(
        "--open_eps",
        type=float,
        default=0.0025,
        help="epsilon tilt for midpoint pricing in continuous session (e.g., 0.0025 for 0.25%)",
    )
    p.add_argument("--account-id", default=None, help="override GM account id (env GM_ACCOUNT_ID otherwise)")
    p.add_argument("--trade-cal", default=None, help="optional Tushare trade calendar CSV (columns cal_date,is_open) to skip non-trading days")
    return p.parse_args()


def _tz_shanghai():
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo not available; use Python 3.9+ or install tzdata.")
    return ZoneInfo("Asia/Shanghai")


def _parse_time_str(tstr: str) -> time:
    parts = [int(x) for x in tstr.strip().split(":")]
    if len(parts) == 2:
        return time(parts[0], parts[1], 0)
    if len(parts) == 3:
        return time(parts[0], parts[1], parts[2])
    raise ValueError(f"Invalid time format: {tstr}")


def _build_today_schedule(times_str: str, now_sh: datetime, open_dates: Optional[Set[str]] = None) -> List[datetime]:
    sh = _tz_shanghai()
    times = [t for t in (s.strip() for s in times_str.split(",")) if t]
    sched: List[datetime] = []
    for ts in times:
        tt = _parse_time_str(ts)
        dt = datetime(now_sh.year, now_sh.month, now_sh.day, tt.hour, tt.minute, tt.second, tzinfo=sh)
        if dt <= now_sh:
            dt = dt + timedelta(days=1)
        if open_dates:
            guard = 0
            while dt.strftime("%Y%m%d") not in open_dates:
                dt = dt + timedelta(days=1)
                guard += 1
                if guard > 366:
                    raise RuntimeError("trade calendar appears to contain no open days within a year; please verify")
        sched.append(dt)
    sched.sort()
    return sched


def _is_in_auction_window(t_sh: time) -> bool:
    return time(9, 15, 0) <= t_sh <= time(9, 25, 0)


def _should_use_second_slice(t_sh: time, second_slice_times: List[time]) -> bool:
    for tt in second_slice_times:
        if t_sh.hour == tt.hour and t_sh.minute == tt.minute and t_sh.second == tt.second:
            return True
    return False


def _run_once(cfg: dict, logs_dir: str, csv_path: str, src: str, place: bool, override_buy: bool, auction_mode: bool, open_eps: float, max_slices_open: int, config_path: str = None, account_id: str = None) -> None:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_trading_once.py"),
        "--csv", csv_path,
        "--src", src,
        "--open_eps", str(float(open_eps)),
        "--max_slices_open", str(int(max_slices_open)),
    ]
    if config_path:
        cmd += ["--config", config_path]
    if account_id:
        cmd += ["--account-id", account_id]
    if auction_mode:
        cmd.append("--auction-mode")
    if place:
        cmd.append("--place")
    if override_buy:
        cmd.append("--override_buy")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path = os.path.join(logs_dir, "scheduler.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[UTC {datetime.utcnow().isoformat()}] CMD={' '.join(cmd)}\n")
        f.write(proc.stdout + "\n")
        f.write(f"RET={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Scheduled run failed with code {proc.returncode}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logs_dir = ensure_logs_dir(cfg)

    sh = _tz_shanghai()
    second_slice_times = [_parse_time_str(s) for s in [t.strip() for t in args.second_slice_times.split(",") if t.strip()]]

    open_dates: Optional[Set[str]] = None
    if args.trade_cal:
        cal_path = os.path.abspath(args.trade_cal)
        if not os.path.exists(cal_path):
            raise FileNotFoundError(f"trade calendar not found: {cal_path}")
        open_dates = set()
        with open(cal_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cal_date = (row.get("cal_date") or "").strip().strip('"')
                is_open = (row.get("is_open") or "0").strip().strip('"')
                if cal_date and is_open == "1":
                    open_dates.add(cal_date)
        if not open_dates:
            raise RuntimeError(f"trade calendar {cal_path} contains no open trading days")

    while True:
        now_sh = datetime.now(sh)
        schedule = _build_today_schedule(args.times, now_sh, open_dates=open_dates)
        for dt in schedule:
            # sleep until dt (convert using Shanghai tz naive difference)
            while True:
                now_sh = datetime.now(sh)
                if now_sh >= dt:
                    break
                delta = (dt - now_sh).total_seconds()
                sleep_s = max(1.0, min(60.0, delta))
                time_mod.sleep(sleep_s)

            # decide mode by time-of-day in Beijing
            t_sh = dt.time()
            auction_mode = _is_in_auction_window(t_sh)
            max_slices_open = 2 if _should_use_second_slice(t_sh, second_slice_times) else 1

            try:
                _run_once(
                    cfg=cfg,
                    logs_dir=logs_dir,
                    csv_path=args.csv,
                    src=args.src,
                    place=bool(args.place),
                    override_buy=bool(args.override_buy),
                    auction_mode=bool(auction_mode),
                    open_eps=float(args.open_eps),
                    max_slices_open=int(max_slices_open),
                    config_path=args.config,
                    account_id=args.account_id,
                )
            except Exception as e:  # noqa: E722
                # log and continue to next event
                log_path = os.path.join(logs_dir, "scheduler.log")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[UTC {datetime.utcnow().isoformat()}] ERROR: {e}\n\n")

        if open_dates:
            # advance now_sh to next trading day start to rebuild schedule
            guard = 0
            next_day = datetime.now(sh) + timedelta(days=1)
            while next_day.strftime("%Y%m%d") not in open_dates:
                next_day = next_day + timedelta(days=1)
                guard += 1
                if guard > 366:
                    raise RuntimeError("trade calendar iteration failed: cannot locate next trading day")
            # sleep until 06:00 Beijing time on next trading day to rebuild schedule
            next_anchor = datetime(next_day.year, next_day.month, next_day.day, 6, 0, 0, tzinfo=sh)
            while True:
                now_sh = datetime.now(sh)
                if now_sh >= next_anchor:
                    break
                delta = (next_anchor - now_sh).total_seconds()
                sleep_s = max(60.0, min(600.0, delta))
                time_mod.sleep(sleep_s)

        # after finishing today's (or upcoming next-day) schedule, loop back to compute next


if __name__ == "__main__":
    main()


