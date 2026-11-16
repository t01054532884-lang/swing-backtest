"""All indicators use CLOSED bars only; periods always refer to the chosen interval."""

from .models import Candle


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def wilder(values: list[float], period: int) -> float:
    if len(values) < period:
        raise ValueError("Insufficient history")
    result = sum(values[:period]) / period
    for value in values[period:]:
        result = ((period - 1) * result + value) / period
    return result


def features(bars: list[Candle], min_count=121) -> dict:
    if len(bars) < max(21, min_count):
        raise ValueError(f"At least {max(21, min_count)} closed bars required")
    close = [float(b.close) for b in bars]
    changes = [b - a for a, b in zip(close, close[1:])]
    gain = wilder([max(x, 0) for x in changes], 14)
    loss = wilder([max(-x, 0) for x in changes], 14)
    rsi = 50 if gain == loss == 0 else 100 if loss == 0 else 100 - 100 / (1 + gain / loss)
    tr = [
        max(float(b.high - b.low), abs(float(b.high) - close[i - 1]), abs(float(b.low) - close[i - 1]))
        for i, b in enumerate(bars)
        if i
    ]
    fast, slow = ema(close, 12), ema(close, 26)
    macd = [a - b for a, b in zip(fast, slow)]
    volumes = [float(b.volume) for b in bars[-21:-1]]
    average_volume = sum(volumes) / len(volumes)
    e20, e60, e120 = (ema(close, p) for p in (20, 60, 120))
    return {
        "close": close[-1],
        "ema20": e20[-1],
        "ema60": e60[-1],
        "ema120": e120[-1],
        "previous_close": close[-2],
        "previous_ema20": e20[-2],
        "rsi14": rsi,
        "atr14": wilder(tr, 14),
        "macd": macd[-1],
        "macd_signal": ema(macd, 9)[-1],
        "volume_ratio": float(bars[-1].volume) / average_volume if average_volume else 0,
        "support20": min(float(b.low) for b in bars[-21:-1]),
        "resistance20": max(float(b.high) for b in bars[-21:-1]),
    }
