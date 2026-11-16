"""데이터 읽기와 원본 규칙 코드 불러오기."""

import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "vendor"))

from astra_trader.models import Candle  # noqa: E402
from astra_trader.swing_strategy import TIMEFRAMES  # noqa: E402

STEP = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
WINDOW = 500  # 원본 알림 서비스가 시간봉마다 가져오는 봉 개수 (market.candles(limit=500))
RULES_FIXED_MS = 1_790_553_600_000  # 2026-09-28 00:00 UTC, 규칙을 마지막으로 고친 날


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def load(symbol, interval):
    df = pd.read_csv(DATA / f"{symbol}_{interval}.csv.gz")
    df = df[df.close_time < config()["data_end_ms"]].reset_index(drop=True)
    gaps = df.open_time.diff().dropna()
    if (gaps != STEP[interval]).any():
        raise ValueError(f"{symbol} {interval}: 빠진 봉 {int((gaps != STEP[interval]).sum())}곳")
    return df


def load_funding(symbol):
    path = DATA / f"{symbol}_funding.csv.gz"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["time", "rate"])


def candles(df):
    """원본 코드가 받는 Candle 객체 목록 (가격은 Decimal)."""
    return [
        Candle(open_time=int(r.open_time), close_time=int(r.close_time), open=Decimal(repr(r.open)),
               high=Decimal(repr(r.high)), low=Decimal(repr(r.low)), close=Decimal(repr(r.close)),
               volume=Decimal(repr(r.volume)))
        for r in df.itertuples()
    ]


def ema(values, period):
    """원본 indicators.ema와 같은 식 (첫 값에서 시작)."""
    alpha = 2 / (period + 1)
    out = np.empty(len(values))
    acc = values[0]
    for i, v in enumerate(values):
        acc = v if i == 0 else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out
