"""바이낸스 USD-M 공개 시세(키 없음)를 내려받는다: 4개 시간봉 + 펀딩비.

python -X utf8 src/fetch.py            # config.json의 모든 종목
python -X utf8 src/fetch.py BTCUSDT    # 한 종목만

결과: data/<SYMBOL>_<interval>.csv.gz, data/<SYMBOL>_funding.csv.gz
이미 받은 파일이 있으면 마지막 봉 다음부터 이어 받는다.
"""

import gzip
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://fapi.binance.com"
STEP = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]


def get(client, path, params):
    for attempt in range(5):
        r = client.get(path, params=params)
        if r.status_code == 200:
            used = int(r.headers.get("x-mbx-used-weight-1m", "0"))
            if used > 1800:
                time.sleep(30)
            return r.json()
        if r.status_code in (418, 429):
            time.sleep(60)
            continue
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()


def klines(client, symbol, interval, start, end):
    rows = []
    while start < end:
        batch = get(client, "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "startTime": start, "endTime": end, "limit": 1500})
        if not batch:
            break
        rows += [[int(b[0]), b[1], b[2], b[3], b[4], b[5], int(b[6])] for b in batch]
        start = int(batch[-1][0]) + STEP[interval]
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=COLS)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def funding(client, symbol, start, end):
    rows = []
    while start < end:
        batch = get(client, "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start, "endTime": end, "limit": 1000})
        if not batch:
            break
        rows += [[int(b["fundingTime"]), float(b["fundingRate"])] for b in batch]
        start = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.15)
    return pd.DataFrame(rows, columns=["time", "rate"])


def merge_save(path, new, key):
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates(key, keep="last")
    new = new.sort_values(key).reset_index(drop=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        new.to_csv(f, index=False)
    return new


def main():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    symbols = sys.argv[1:] or cfg["symbols"]
    end = cfg["data_end_ms"]
    DATA.mkdir(exist_ok=True)
    with httpx.Client(base_url=BASE, timeout=30) as client:
        info = {s["symbol"]: s for s in get(client, "/fapi/v1/exchangeInfo", {})["symbols"]}
        for symbol in symbols:
            start0 = max(cfg["data_start_ms"], info[symbol]["onboardDate"])
            for interval in STEP:
                path = DATA / f"{symbol}_{interval}.csv.gz"
                start = start0
                if path.exists():
                    last = pd.read_csv(path, usecols=["open_time"])["open_time"].max()
                    start = int(last) + STEP[interval]
                new = klines(client, symbol, interval, start, end)
                df = merge_save(path, new, "open_time")
                # 끝나지 않은 봉은 버린다 (close_time이 받은 끝 시각보다 뒤)
                print(f"{symbol} {interval}: {len(df):,}봉")
            path = DATA / f"{symbol}_funding.csv.gz"
            start = start0
            if path.exists():
                start = int(pd.read_csv(path)["time"].max()) + 1
            df = merge_save(path, funding(client, symbol, start, end), "time")
            print(f"{symbol} funding: {len(df):,}건")


if __name__ == "__main__":
    main()
