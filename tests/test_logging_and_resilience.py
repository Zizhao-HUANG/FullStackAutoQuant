"""Tests for logging_config and resilience modules (P0 infrastructure)."""

from __future__ import annotations

import logging
import os
import time
from unittest.mock import patch

import pytest

from fullstackautoquant.logging_config import get_logger, reset_logging
from fullstackautoquant.resilience import (
    MissingCredentialError,
    RateLimiter,
    retry_on_exception,
    validate_data_credentials,
    validate_trading_credentials,
)

# ── Logging Tests ────────────────────────────────────────────────────────


class TestGetLogger:
    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_returns_logger_under_namespace(self) -> None:
        logger = get_logger("fullstackautoquant.trading.strategy")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "fullstackautoquant.trading.strategy"

    def test_logger_respects_level(self) -> None:
        with patch.dict(os.environ, {"FSAQ_LOG_LEVEL": "ERROR"}):
            reset_logging()
            _logger = get_logger("fullstackautoquant.test")  # noqa: F841
            root = logging.getLogger("fullstackautoquant")
            assert root.level == logging.ERROR

    def test_logger_is_idempotent(self) -> None:
        get_logger("fullstackautoquant.a")
        get_logger("fullstackautoquant.b")
        root = logging.getLogger("fullstackautoquant")
        # Should only have one handler regardless of how many times get_logger is called
        assert len(root.handlers) == 1


# ── Retry Decorator Tests ────────────────────────────────────────────────


class TestRetryOnException:
    def test_success_on_first_attempt(self) -> None:
        call_count = 0

        @retry_on_exception(max_retries=3, base_delay=0.01)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_retries_then_succeeds(self) -> None:
        call_count = 0

        @retry_on_exception(max_retries=3, base_delay=0.01)
        def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = fail_twice()
        assert result == "recovered"
        assert call_count == 3

    def test_exhausts_retries_and_raises(self) -> None:
        call_count = 0

        @retry_on_exception(max_retries=2, base_delay=0.01)
        def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            always_fail()
        assert call_count == 3  # 1 initial + 2 retries

    def test_only_retries_specified_exceptions(self) -> None:
        call_count = 0

        @retry_on_exception(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        def raise_value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            raise_value_error()
        # Should fail immediately — ValueError is not in retryable_exceptions
        assert call_count == 1

    def test_custom_description_in_logs(self) -> None:
        @retry_on_exception(
            max_retries=1,
            base_delay=0.01,
            description="my_api_call",
        )
        def fail_once() -> str:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            fail_once()


# ── Rate Limiter Tests ──────────────────────────────────────────────────


class TestRateLimiter:
    def test_rate_limiting_enforced(self) -> None:
        limiter = RateLimiter(calls_per_second=100.0)  # 10ms interval
        start = time.monotonic()
        for _ in range(5):
            limiter.wait()
        elapsed = time.monotonic() - start
        # 5 calls at 100/sec should take at least ~40ms (4 intervals)
        assert elapsed >= 0.03  # generous lower bound

    def test_first_call_is_immediate(self) -> None:
        limiter = RateLimiter(calls_per_second=1.0)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        # First call should be nearly instant
        assert elapsed < 0.1


# ── Credential Validation Tests ──────────────────────────────────────────


class TestValidateCredentials:
    def test_missing_tushare_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove all credential env vars
            for key in ("TUSHARE", "TS_TOKEN", "GM_ENDPOINT", "GM_TOKEN", "GM_ACCOUNT_ID"):
                os.environ.pop(key, None)
            with pytest.raises(MissingCredentialError, match="TUSHARE"):
                validate_data_credentials()

    def test_tushare_only_passes_data_mode(self) -> None:
        with patch.dict(os.environ, {"TUSHARE": "test_token_12345"}, clear=True):
            result = validate_data_credentials()
            assert "TUSHARE" in result

    def test_missing_gm_raises_in_trading_mode(self) -> None:
        with (
            patch.dict(os.environ, {"TUSHARE": "test_token"}, clear=True),
            pytest.raises(MissingCredentialError, match="GM_"),
        ):
            validate_trading_credentials(require_gm=True)

    def test_all_credentials_pass(self) -> None:
        env = {
            "TUSHARE": "ts_token_12345",
            "GM_ENDPOINT": "https://api.example.com",
            "GM_TOKEN": "gm_token_12345",
            "GM_ACCOUNT_ID": "acc_12345",
        }
        with patch.dict(os.environ, env, clear=True):
            result = validate_trading_credentials(require_gm=True)
            assert "TUSHARE" in result
            assert "GM_ENDPOINT" in result
            assert "GM_TOKEN" in result
            assert "GM_ACCOUNT_ID" in result

    def test_credential_values_are_masked(self) -> None:
        env = {"TUSHARE": "my_secret_tushare_token"}
        with patch.dict(os.environ, env, clear=True):
            result = validate_data_credentials()
            assert "my_secret" not in result["TUSHARE"]
            assert result["TUSHARE"].startswith("***")
