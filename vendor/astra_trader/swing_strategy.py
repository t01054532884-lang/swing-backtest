"""Closed-candle swing rules. Scores are evidence counts, never probabilities."""

from decimal import Decimal

from pydantic import Field

from .backtest import validate_bars
from .binance import INTERVAL_MS
from .indicators import ema, features
from .models import D, StrictModel

TIMEFRAMES = ("15m", "1h", "4h", "1d")


class SwingSettings(StrictModel):
    symbols: list[str] = Field(min_length=1, max_length=100)
    account_loss_limit: Decimal = Field(default=D("0.20"), gt=0, le=D("0.20"))
    price_stop_limit: Decimal = Field(default=D("0.10"), gt=0, le=D("0.10"))
    # Conservative implementation defaults, not user-approved trading allocations.
    symbol_risk_fraction: Decimal = Field(default=D("0.01"), gt=0, le=D("0.02"))
    leverage: int = Field(default=3, ge=3, le=4)
    cooldown_seconds: int = Field(default=14400, ge=900)
    account_poll_seconds: int = Field(default=15, ge=10, le=60)
    max_hold_days: int = Field(default=21, ge=2, le=21)
    fee_buffer: Decimal = Field(default=D("0.002"), ge=D("0.001"), le=D("0.01"))
    # User-approved "5 of 7" (71%); with the 8th check (4h 이평 정배열) the equivalent is 6 of 8.
    min_score: int = Field(default=6, ge=5, le=8)
    # Stop below support / above resistance; recommend only if the nearest opposing level pays this multiple.
    min_reward_risk: Decimal = Field(default=D("1.5"), ge=1, le=5)
    # A high aggregate score must not create an entry without an actual setup.
    require_hour_trend: bool = True
    require_entry_trigger: bool = True
    option_underlyings: dict[str, str] = Field(default_factory=dict)
    options_cache_seconds: int = Field(default=1800, ge=900, le=14400)

    def checked_symbols(self):
        if len(set(self.symbols)) != len(self.symbols) or any(
            not s.isalnum() or not s.isupper() or not s.endswith("USDT") for s in self.symbols
        ):
            raise ValueError("Unique uppercase USDT symbols required")
        if any(
            symbol not in self.symbols or not ticker.isalnum()
            for symbol, ticker in self.option_underlyings.items()
        ):
            raise ValueError("Option mappings must use configured symbols and alphanumeric US tickers")
        return self.symbols


def adx(bars, period=14):
    tr, plus, minus = [], [], []
    for a, b in zip(bars, bars[1:]):
        up, down = float(b.high - a.high), float(a.low - b.low)
        plus.append(up if up > down and up > 0 else 0)
        minus.append(down if down > up and down > 0 else 0)
        tr.append(float(max(b.high - b.low, abs(b.high - a.close), abs(b.low - a.close))))
    averages = [sum(v[:period]) / period for v in (tr, plus, minus)]
    dx = []
    for i in range(period - 1, len(tr)):
        if i >= period:
            averages = [(a * (period - 1) + v[i]) / period for a, v in zip(averages, (tr, plus, minus))]
        _, p, m = averages
        dx.append(100 * abs(p - m) / (p + m) if p + m else 0)
    if len(dx) < period:
        raise ValueError("ADX warmup missing")
    value = sum(dx[:period]) / period
    for x in dx[period:]:
        value = (value * (period - 1) + x) / period
    return value


MA_PERIODS = (5, 25, 60, 120)  # User-specified chart lines (Korean HTS standard, simple averages)
CROSS_PAIRS = ((5, 25), (25, 60), (60, 120))
CROSS_LOOKBACK = 5  # A cross within the last 5 closed candles counts as "recent"


def sma_series(values, n):
    if len(values) < n:
        return []
    total = sum(values[:n])
    out = [total / n]
    for i in range(n, len(values)):
        total += values[i] - values[i - n]
        out.append(total / n)
    return out


def last_cross(fast, slow):
    """(direction, bars ago) of the latest fast/slow cross; direction 1 = golden, -1 = dead."""
    length = min(len(fast), len(slow))
    fast, slow = fast[-length:], slow[-length:]
    for ago in range(1, length):
        before = fast[-ago - 1] - slow[-ago - 1]
        after = fast[-ago] - slow[-ago]
        if before <= 0 < after:
            return 1, ago - 1
        if before >= 0 > after:
            return -1, ago - 1
    return 0, None


def swing_points(bars, width=3, lookback=60):
    """Confirmed pivot lows/highs: extreme of the surrounding `width` candles on each side."""
    recent = bars[-lookback:]
    lows, highs = [], []
    for i in range(width, len(recent) - width):
        window = recent[i - width : i + width + 1]
        if recent[i].low == min(b.low for b in window):
            lows.append(float(recent[i].low))
        if recent[i].high == max(b.high for b in window):
            highs.append(float(recent[i].high))
    return lows, highs


def frame_features(bars, daily=False):
    f = features(bars, min_count=61 if daily else 201)
    close = [float(b.close) for b in bars]
    averages = {n: sma_series(close, n) for n in MA_PERIODS}
    ma = {n: (series[-1] if series else None) for n, series in averages.items()}
    fast, slow = ema(close, 20), ema(close, 50)
    crosses = {}
    for a, b in CROSS_PAIRS:
        direction, ago = last_cross(averages[a], averages[b]) if averages[b] else (0, None)
        crosses[f"{a}/{b}"] = {"direction": direction, "ago": ago}
    values = [ma[n] for n in MA_PERIODS if ma[n] is not None]
    lows, highs = swing_points(bars)
    f.update(
        ma5=ma[5],
        ma25=ma[25],
        ma60=ma[60],
        ma120=ma[120],
        previous_ma25=averages[25][-2],
        crosses=crosses,
        # 정배열 5>25>60>120 (1), 역배열 (-1); daily with <120 candles compares the lines it has.
        alignment=1
        if all(x > y for x, y in zip(values, values[1:]))
        else -1
        if all(x < y for x, y in zip(values, values[1:]))
        else 0,
        swing_lows=lows,
        swing_highs=highs,
        ema50=slow[-1],
        adx=adx(bars),
        golden=fast[-2] <= slow[-2] and fast[-1] > slow[-1],
        dead=fast[-2] >= slow[-2] and fast[-1] < slow[-1],
        trend_reference_period=60,
    )
    if daily:
        f.pop("ema120")
        f.pop("ema60")
    # Trend = price vs the 60 line plus 25/60 order (golden/dead-cross state).
    f["trend"] = (
        1 if f["close"] > ma[60] and ma[25] > ma[60] else -1 if f["close"] < ma[60] and ma[25] < ma[60] else 0
    )
    return f


def analyze_frames(frames, at, min_score=7):
    result = {}
    for interval in TIMEFRAMES:
        # Explicit as-of filtering prevents leakage from higher timeframe future closes.
        bars = [b for b in frames[interval] if b.close_time <= at]
        minimum = 61 if interval == "1d" else 201
        if len(bars) < minimum:
            raise ValueError(f"{interval}: need at least {minimum} completed candles")
        validate_bars(bars, interval, min_count=minimum)
        if at - bars[-1].close_time >= INTERVAL_MS[interval]:
            raise ValueError(f"{interval}: missing latest completed candle")
        result[interval] = frame_features(bars, daily=interval == "1d")
    daily, four, hour, fast = (result[x] for x in ("1d", "4h", "1h", "15m"))
    side = daily["trend"] if daily["trend"] == four["trend"] else 0
    momentum = (40 <= hour["rsi14"] <= 68) if side == 1 else (32 <= hour["rsi14"] <= 60)
    macd = (hour["macd"] > hour["macd_signal"]) if side == 1 else (hour["macd"] < hour["macd_signal"])
    cross = fast["crosses"]["5/25"]
    trigger = (
        # 15m 5/25 golden (dead) cross within the last 2 candles, or a close back over (under) the 25 line.
        (cross["direction"] == side and cross["ago"] is not None and cross["ago"] <= 1)
        or (
            side * (fast["previous_close"] - fast["previous_ma25"]) <= 0
            and side * (fast["close"] - fast["ma25"]) > 0
        )
        if side != 0
        else False
    )
    checks = {
        "상위 추세 일치": side != 0,
        "1시간 추세 일치": side != 0 and hour["trend"] == side,
        "4시간 이평 정배열": side != 0 and four["alignment"] == side,
        "4시간 ADX ≥ 25": four["adx"] >= 25,
        "RSI 과열 필터": momentum,
        "MACD 방향": macd,
        "15분 교차/눌림 회복": trigger,
        "거래량 ≥ 평균 1.2배": fast["volume_ratio"] >= 1.2,
    }
    score = sum(checks.values())
    # Direction comes only from daily/4h agreement, so a partial score never invents a side.
    # Keep the evidence-only action for compatibility; the live runtime always applies
    # the mandatory gates below before an alert can be emitted.
    action = ("LONG" if side == 1 else "SHORT") if side != 0 and score >= min_score else "WAIT"
    return {
        "action": action,
        "frames": result,
        "checks": checks,
        "score": score,
        "at": at,
        "side": side,
    }


def apply_entry_gates(analysis, settings):
    """Turn evidence into an action only after the non-negotiable setup gates pass."""
    checks = analysis["checks"]
    gates = {
        "상위 추세 일치": checks["상위 추세 일치"],
        "1시간 추세 필수": not settings.require_hour_trend or checks["1시간 추세 일치"],
        "15분 진입 트리거 필수": not settings.require_entry_trigger or checks["15분 교차/눌림 회복"],
    }
    side = analysis["side"]
    action = (
        ("LONG" if side == 1 else "SHORT")
        if side != 0 and analysis["score"] >= settings.min_score and all(gates.values())
        else "WAIT"
    )
    return {**analysis, "action": action, "gates": gates}


CHECK_COUNT = 8


def strength(score):
    return "강" if score >= 8 else "중" if score == 7 else "약"


def scale_levels(entry, stop):
    """1st at entry, add-ons at 1/3 and 2/3 of the way to the unchanged stop."""
    distance = entry - stop
    return [entry, entry - distance / 3, entry - distance * 2 / 3]


def split_plan(entry, stop, budget, fee, cap_notional, weights=(D("0.4"), D("0.3"), D("0.3"))):
    """Size 40/30/30 tranches so that, if all fill and the stop hits, loss ≈ budget."""
    levels = scale_levels(entry, stop)
    loss_per_notional = sum(w * (abs(p - stop) / p + fee) for w, p in zip(weights, levels))
    total = min(max(D(0), budget) / loss_per_notional, max(D(0), cap_notional))
    return [{"price": p, "notional": total * w, "quantity": total * w / p} for w, p in zip(weights, levels)]


MA_LEVELS = (("1h", 60), ("1h", 120), ("4h", 60), ("4h", 120), ("1d", 25), ("1d", 60), ("1d", 120))
FRAME_NAME = {"15m": "15분", "1h": "1시간", "4h": "4시간", "1d": "일봉"}


def chart_levels(analysis):
    """Candidate support/resistance: confirmed 1h/4h/daily pivots and the 60/120 (daily 25/60/120) lines."""
    levels = []
    for interval in ("1h", "4h", "1d"):
        f = analysis["frames"][interval]
        name = FRAME_NAME[interval]
        levels += [(D(x), f"{name} 저점") for x in f.get("swing_lows", [])]
        levels += [(D(x), f"{name} 고점") for x in f.get("swing_highs", [])]
    for interval, period in MA_LEVELS:
        value = analysis["frames"][interval].get(f"ma{period}")
        if value:
            levels.append((D(value), f"{FRAME_NAME[interval]} {period}선"))
    return levels


def structure_levels(analysis, price, settings):
    """Stop just beyond the nearest real support (long) / resistance (short); targets at opposing levels.

    Falls back to the daily-volatility stop when no level lies within the 10% rule. The stop keeps
    at least one 1h ATR of room so ordinary noise does not trigger it.
    """
    side = 1 if analysis["action"] == "LONG" else -1
    frames = analysis["frames"]
    atr_1h, atr_4h = D(frames["1h"]["atr14"]), D(frames["4h"]["atr14"])
    limit = price * settings.price_stop_limit
    levels = chart_levels(analysis)
    # Same-side levels (support for longs) sorted nearest first; opposing levels likewise.
    behind = sorted(
        ((p, n) for p, n in levels if side * (price - p) >= atr_1h), key=lambda x: side * (price - x[0])
    )
    ahead = sorted(
        ((p, n) for p, n in levels if side * (p - price) > atr_1h / 4), key=lambda x: side * (x[0] - price)
    )
    anchor = next(((p, n) for p, n in behind if side * (price - p) + atr_1h / 2 <= limit), None)
    if anchor:
        stop = anchor[0] - side * atr_1h / 2
    else:
        swing = max(D(frames["1d"]["atr14"]) * D("1.5"), atr_4h * 3)
        stop = price - side * min(swing, limit)
    risk = side * (price - stop)
    if risk <= 0:
        raise ValueError("Stop distance must be positive")
    resistance = ahead[0] if ahead else None
    reward_risk = side * (resistance[0] - price) / risk if resistance else None
    if resistance:
        further = [p for p, _ in ahead[1:] if side * (p - resistance[0]) >= risk / 2]
        targets = [resistance[0], further[0] if further else resistance[0] + side * risk]
    else:
        targets = [price + side * risk * 2, price + side * risk * 3]  # no overhead level: open range
    if min(stop, *targets) <= 0:
        raise ValueError("Invalid price levels")
    return {
        "stop": stop,
        "targets": targets,
        "support": anchor,
        "resistance": resistance,
        "reward_risk": reward_risk,
        "blocked": (
            f"{'저항' if side > 0 else '지지'} 근접 · 손익비 {reward_risk:.1f} < {settings.min_reward_risk}"
            if reward_risk is not None and reward_risk < settings.min_reward_risk
            else None
        ),
    }


def entry_levels(analysis, price, settings):
    levels = structure_levels(analysis, price, settings)
    return levels["stop"], levels["targets"]


def position_levels(entry, sign, settings, analysis=None):
    """User rule for held positions: stop 10% from average; take profit 1/3 at +10% and +20%.

    Chart damage (trend or support break) is reported separately as an early-exit review,
    so the hard stop is never tightened to a volatility band (that produced 1% stops).
    """
    limit = settings.price_stop_limit
    stop = entry * (1 - sign * limit)
    return stop, [entry * (1 + sign * limit * k) for k in (1, 2)]
