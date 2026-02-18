"""Retry and resilience utilities for external API calls.

Provides decorators and helpers for:
- Retrying transient failures with exponential backoff (Tushare, GM Trade)
- Rate limiting for API compliance
- Startup credential validation (fail-fast before market open)
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from fullstackautoquant.logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Retry decorator (no external dependency on tenacity)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0  # seconds
_DEFAULT_MAX_DELAY = 30.0  # seconds cap


def retry_on_exception(
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _DEFAULT_BASE_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    description: str = "",
) -> Callable[[F], F]:
    """Decorator that retries a function on transient failures with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds between retries.
        max_delay: Cap on delay between retries.
        retryable_exceptions: Tuple of exception types that trigger a retry.
        description: Human-readable label for log messages.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            label = description or func.__qualname__
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 2):  # attempt 1 = first call
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt > max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            label,
                            attempt,
                            exc,
                            exc_info=True,
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(
                        "%s attempt %d/%d failed (%s), retrying in %.1fs…",
                        label,
                        attempt,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            # Should not reach here, but satisfy type checker
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket style)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple token-bucket rate limiter for API calls.

    Usage:
        limiter = RateLimiter(calls_per_second=2)
        for batch in batches:
            limiter.wait()
            api_call(batch)
    """

    def __init__(self, calls_per_second: float = 2.0) -> None:
        self._min_interval = 1.0 / max(calls_per_second, 0.01)
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Startup secrets validation
# ---------------------------------------------------------------------------


class MissingCredentialError(RuntimeError):
    """Raised when a required credential is not configured."""


def validate_trading_credentials(*, require_gm: bool = True) -> dict[str, str]:
    """Validate that all required credentials are present before trading starts.

    Call this ONCE at startup, before any trading logic runs.
    Fails fast with a clear error if any credential is missing.

    Args:
        require_gm: If True (default for live trading), require GM Trade
                    credentials. Set to False for signal-only / backtest modes.

    Returns:
        Dict of validated credential keys and their (masked) status.

    Raises:
        MissingCredentialError: If any required credential is missing.
    """
    from dotenv import load_dotenv

    load_dotenv()

    results: dict[str, str] = {}
    missing: list[str] = []

    # Tushare token (always required for market data)
    tushare = os.getenv("TUSHARE") or os.getenv("TS_TOKEN")
    if tushare:
        results["TUSHARE"] = f"***{tushare[-4:]}" if len(tushare) > 4 else "***"
    else:
        missing.append("TUSHARE (or TS_TOKEN)")

    if require_gm:
        gm_fields = {
            "GM_ENDPOINT": os.getenv("GM_ENDPOINT"),
            "GM_TOKEN": os.getenv("GM_TOKEN"),
            "GM_ACCOUNT_ID": os.getenv("GM_ACCOUNT_ID"),
        }
        for key, val in gm_fields.items():
            if val:
                results[key] = f"***{val[-4:]}" if len(val) > 4 else "***"
            else:
                missing.append(key)

    if missing:
        msg = (
            "Missing required credentials. Set these in your .env file or environment:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nSee .env.example for reference."
        )
        logger.error("Credential validation failed: %s", ", ".join(missing))
        raise MissingCredentialError(msg)

    logger.info("Credential validation passed: %s", ", ".join(results.keys()))
    return results


def validate_data_credentials() -> dict[str, str]:
    """Validate credentials for data/signal-only mode (no GM Trade required)."""
    return validate_trading_credentials(require_gm=False)
