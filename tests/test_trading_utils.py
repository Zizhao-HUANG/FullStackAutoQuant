"""Tests for fullstackautoquant.trading.utils — symbol conversion, pricing, lot sizing."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import yaml

from fullstackautoquant.trading.utils import (
    board_limit_ratio_for_symbol,
    clamp_volume_to_lot,
    compute_allowed_price,
    compute_auction_price,
    compute_limit_price_from_rt_preclose,
    compute_manual_price,
    compute_open_mid_price,
    ensure_logs_dir,
    gm_to_instrument,
    gm_to_ts_code,
    instrument_to_gm,
    load_config,
    load_json,
    max_order_volume_for_symbol,
    min_order_lot_for_symbol,
    round_price,
    save_json,
    ts_code_to_gm,
)


# ── Symbol Conversion ────────────────────────────────────────────────


class TestInstrumentToGm:
    def test_sh_instrument(self):
        assert instrument_to_gm("SH600000") == "SHSE.600000"

    def test_sz_instrument(self):
        assert instrument_to_gm("SZ000001") == "SZSE.000001"

    def test_lowercase_sh(self):
        assert instrument_to_gm("sh600000") == "SHSE.600000"

    def test_invalid_length(self):
        assert instrument_to_gm("SH60") is None

    def test_invalid_market(self):
        assert instrument_to_gm("HK600000") is None

    def test_non_digit_code(self):
        assert instrument_to_gm("SHABCDEF") is None

    def test_whitespace_stripped(self):
        assert instrument_to_gm("  SH600000  ") == "SHSE.600000"


class TestGmToInstrument:
    def test_shse(self):
        assert gm_to_instrument("SHSE.600000") == "SH600000"

    def test_szse(self):
        assert gm_to_instrument("SZSE.000001") == "SZ000001"

    def test_invalid_prefix(self):
        assert gm_to_instrument("HKEX.00700") is None

    def test_whitespace(self):
        assert gm_to_instrument("  SHSE.600000  ") == "SH600000"


class TestGmToTsCode:
    def test_shse(self):
        assert gm_to_ts_code("SHSE.600000") == "600000.SH"

    def test_szse(self):
        assert gm_to_ts_code("SZSE.000001") == "000001.SZ"

    def test_unknown(self):
        assert gm_to_ts_code("HKEX.00700") is None


class TestTsCodeToGm:
    def test_sh(self):
        assert ts_code_to_gm("600000.SH") == "SHSE.600000"

    def test_sz(self):
        assert ts_code_to_gm("000001.SZ") == "SZSE.000001"

    def test_unknown(self):
        assert ts_code_to_gm("00700.HK") is None

    def test_whitespace(self):
        assert ts_code_to_gm("  600000.SH  ") == "SHSE.600000"


# ── Round Price ──────────────────────────────────────────────────────


class TestRoundPrice:
    def test_round_up(self):
        result = round_price(10.03, 0.01, "up")
        assert result == pytest.approx(10.03)

    def test_round_down(self):
        result = round_price(10.037, 0.01, "down")
        assert result == pytest.approx(10.03)

    def test_round_nearest(self):
        result = round_price(10.035, 0.01, "nearest")
        assert result == pytest.approx(10.04)

    def test_zero_tick_falls_back(self):
        result = round_price(10.037, 0, "up")
        assert result == pytest.approx(10.04)

    def test_negative_tick_falls_back(self):
        result = round_price(10.123, -0.01, "up")
        assert result == pytest.approx(10.12)


# ── Compute Allowed Price ────────────────────────────────────────────


class TestComputeAllowedPrice:
    def test_buy_price(self):
        price = compute_allowed_price("BUY", 10.0, 0.02, -0.02, 0.095, 0.01)
        assert price > 0
        assert price <= 10.0 * 1.095  # must not exceed limit

    def test_sell_price(self):
        price = compute_allowed_price("SELL", 10.0, 0.02, -0.02, 0.095, 0.01)
        assert price > 0
        assert price >= 10.0 * (1.0 - 0.095)  # must not be below limit


class TestComputeLimitPriceFromRtPreclose:
    def test_buy_price(self):
        price = compute_limit_price_from_rt_preclose(
            "BUY", rt_price=10.1, pre_close=10.0, buy_offset=0.02,
            sell_offset=-0.02, limit_threshold=0.095, tick=0.01,
        )
        assert price > 0

    def test_sell_price(self):
        price = compute_limit_price_from_rt_preclose(
            "SELL", rt_price=9.9, pre_close=10.0, buy_offset=0.02,
            sell_offset=-0.02, limit_threshold=0.095, tick=0.01,
        )
        assert price > 0


# ── Board Limit Ratio ────────────────────────────────────────────────


class TestBoardLimitRatio:
    def test_star_market(self):
        assert board_limit_ratio_for_symbol("SHSE.688001") == 0.20

    def test_chinext(self):
        assert board_limit_ratio_for_symbol("SZSE.300001") == 0.20

    def test_default(self):
        assert board_limit_ratio_for_symbol("SHSE.600000") == 0.10


# ── Auction / Mid Price ──────────────────────────────────────────────


class TestComputeAuctionPrice:
    def test_buy_default_board(self):
        price = compute_auction_price("BUY", "SHSE.600000", pre_close=10.0, tick=0.01)
        assert price > 0
        assert price <= 10.0 * 1.10 + 0.1  # near limit_up

    def test_sell_default_board(self):
        price = compute_auction_price("SELL", "SHSE.600000", pre_close=10.0, tick=0.01)
        assert price > 0

    def test_buy_star_market(self):
        price = compute_auction_price("BUY", "SHSE.688001", pre_close=50.0, tick=0.01)
        assert price > 0

    def test_zero_pre_close_buy(self):
        price = compute_auction_price("BUY", "SHSE.600000", pre_close=0.0, tick=0.01)
        assert price >= 0

    def test_zero_pre_close_sell(self):
        price = compute_auction_price("SELL", "SHSE.600000", pre_close=0.0, tick=0.01)
        assert price >= 0

    def test_with_limit_overrides(self):
        price = compute_auction_price(
            "BUY", "SHSE.600000", pre_close=10.0, tick=0.01,
            limit_up_override=10.95, limit_down_override=9.05,
        )
        assert price > 0


class TestComputeOpenMidPrice:
    def test_buy_with_bid_ask(self):
        price = compute_open_mid_price(
            "BUY", "SHSE.600000", bid=9.98, ask=10.02,
            rt_price=10.0, pre_close=10.0, eps=0.001, tick=0.01,
        )
        assert price > 0

    def test_sell_with_bid_ask(self):
        price = compute_open_mid_price(
            "SELL", "SHSE.600000", bid=9.98, ask=10.02,
            rt_price=10.0, pre_close=10.0, eps=0.001, tick=0.01,
        )
        assert price > 0

    def test_no_bid_ask_uses_rt_price(self):
        price = compute_open_mid_price(
            "BUY", "SHSE.600000", bid=0, ask=0,
            rt_price=10.0, pre_close=10.0, eps=0.001, tick=0.01,
        )
        assert price > 0

    def test_no_bid_ask_no_rt_uses_preclose(self):
        price = compute_open_mid_price(
            "BUY", "SHSE.600000", bid=0, ask=0,
            rt_price=0, pre_close=10.0, eps=0.001, tick=0.01,
        )
        assert price > 0


# ── Lot / Volume Utilities ───────────────────────────────────────────


class TestLotUtils:
    def test_min_lot_default(self):
        assert min_order_lot_for_symbol("SHSE.600000") == 100

    def test_min_lot_star(self):
        assert min_order_lot_for_symbol("SHSE.688001") == 200

    def test_max_volume_default(self):
        assert max_order_volume_for_symbol("SHSE.600000") == 1_000_000

    def test_max_volume_star(self):
        assert max_order_volume_for_symbol("SHSE.688001") == 100_000

    def test_clamp_positive(self):
        assert clamp_volume_to_lot("SHSE.600000", 350) == 300

    def test_clamp_zero(self):
        assert clamp_volume_to_lot("SHSE.600000", 0) == 0

    def test_clamp_negative(self):
        assert clamp_volume_to_lot("SHSE.600000", -10) == 0


# ── Compute Manual Price ─────────────────────────────────────────────


class TestComputeManualPrice:
    def test_buy(self):
        price = compute_manual_price("BUY", 10.0, 0.02, -0.02, 0.01)
        assert price > 0

    def test_sell(self):
        price = compute_manual_price("SELL", 10.0, 0.02, -0.02, 0.01)
        assert price > 0

    def test_zero_ref_price(self):
        assert compute_manual_price("BUY", 0.0, 0.02, -0.02, 0.01) == 0.0

    def test_zero_tick(self):
        price = compute_manual_price("BUY", 10.0, 0.02, -0.02, 0.0)
        assert price > 0  # should fallback to 0.01

    def test_sell_very_low_price(self):
        price = compute_manual_price("SELL", 0.005, 0.02, -0.02, 0.01)
        assert price > 0  # should be at least tick


# ── JSON / Config Utilities ──────────────────────────────────────────


class TestJsonUtils:
    def test_save_and_load(self, tmp_path):
        data = {"key": "value", "number": 42}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_load_config(self, tmp_path):
        cfg = {
            "portfolio": {"topk": 30},
            "order": {"price_source": "qlib_close"},
            "paths": {"logs_dir": str(tmp_path / "logs")},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
        result = load_config(str(cfg_path))
        assert result["portfolio"]["topk"] == 30

    def test_ensure_logs_dir(self, tmp_path):
        logs = str(tmp_path / "test_logs")
        cfg = {"paths": {"logs_dir": logs}}
        result = ensure_logs_dir(cfg)
        assert result == logs
        assert os.path.isdir(logs)

    def test_ensure_logs_dir_default(self):
        result = ensure_logs_dir({})
        assert isinstance(result, str)
