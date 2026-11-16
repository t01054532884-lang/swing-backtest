"""원본 backtest.py에서 validate_bars만 옮김."""

from .binance import INTERVAL_MS


def validate_bars(bars, interval, min_count=122):
    if len(bars) < min_count:
        raise ValueError(f"Need at least {min_count} bars")
    step = INTERVAL_MS[interval]
    if any(b.close_time != b.open_time + step - 1 for b in bars):
        raise ValueError("Candle length does not match configured interval")
    for a, b in zip(bars, bars[1:]):
        if b.open_time - a.open_time != step or a.close_time != a.open_time + step - 1:
            raise ValueError("Duplicate, missing, unsorted or incorrect-interval candles")


