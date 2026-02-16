import os
import argparse
import subprocess
import sys

from utils import load_config, ensure_logs_dir


def parse_args():
    p = argparse.ArgumentParser(description="Run one-shot trading pipeline")
    p.add_argument("--csv", required=True, help="ranked_scores CSV path")
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--place", action="store_true", help="actually place orders")
    p.add_argument("--override_buy", action="store_true", help="override buy block for this run")
    p.add_argument("--src", default="sina", choices=["sina", "dc"], help="tushare source")
    p.add_argument("--current_positions", default=None, help="optional current positions json path (forwarded to strategy)")
    # passthrough for gm execution behavior
    p.add_argument("--auction-mode", action="store_true", help="use opening auction pricing when within 09:15–09:25 Beijing time")
    p.add_argument("--max_slices_open", type=int, default=1, help="max slices for open execution after 09:30 (default 1; set 2 when condition triggers)")
    p.add_argument("--open_eps", type=float, default=0.0025, help="epsilon tilt for midpoint pricing in continuous session (e.g., 0.0025 for 0.25%)")
    p.add_argument("--account-id", default=None, help="override GM account id (env GM_ACCOUNT_ID otherwise)")
    return p.parse_args()


def run_cmd(cmd: list):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logs_dir = ensure_logs_dir(cfg)

    # 1) signals
    sig_out = os.path.join(logs_dir, "signals_AUTO.json")
    run_cmd([
        sys.executable,
        os.path.join(os.path.dirname(__file__), "signals_from_csv.py"),
        "--csv", args.csv,
        "--config", (args.config or ""),
        "--out", sig_out,
    ])

    # 2) risk
    risk_out = os.path.join(logs_dir, "risk_state_AUTO.json")
    risk_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "risk_manager.py"),
        "--signals", sig_out,
        "--config", (args.config or ""),
        "--out", risk_out,
    ]
    if args.override_buy:
        risk_cmd.append("--override_buy")
    run_cmd(risk_cmd)

    # 3) strategy
    targets_out = os.path.join(logs_dir, "targets_AUTO.json")
    orders_out = os.path.join(logs_dir, "orders_AUTO.json")
    strat_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "strategy_rebalance.py"),
        "--signals", sig_out,
        "--risk_state", risk_out,
        "--config", (args.config or ""),
        "--targets", targets_out,
        "--orders", orders_out,
    ]
    if args.current_positions:
        strat_cmd += ["--current_positions", args.current_positions]
    if args.account_id:
        strat_cmd += ["--account-id", str(args.account_id)]
    run_cmd(strat_cmd)

    # 4) gm place with realtime pricing via tushare
    gm_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "gm_api_wrapper.py"),
        "--orders", orders_out,
        "--config", (args.config or ""),
        "--src", args.src,
    ]
    if args.auction_mode:
        gm_cmd.append("--auction-mode")
    gm_cmd += ["--max_slices_open", str(int(args.max_slices_open)), "--open_eps", str(float(args.open_eps))]
    if args.account_id:
        gm_cmd += ["--account-id", str(args.account_id)]
    if args.place:
        gm_cmd.append("--place")
    run_cmd(gm_cmd)

    # 5) logger/replay
    replay_cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "trade_logger_replay.py"), "--update_nav"]
    if args.config:
        replay_cmd += ["--config", args.config]
    run_cmd(replay_cmd)


if __name__ == "__main__":
    main()
