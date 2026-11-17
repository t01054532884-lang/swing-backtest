"""미국 ETF 일봉(배당·분할 반영)과 기준금리를 받는다 → etf/data/

python -X utf8 etf/fetch_etf.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
DATA = HERE / "data"
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "IEF", "SHY"]
END = "2026-11-17"  # 2026-11-16 종가까지


def main():
    DATA.mkdir(exist_ok=True)
    for t in TICKERS:
        d = yf.download(t, start="2004-01-01", end=END, auto_adjust=True, progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        d.index.name = "date"
        d[["open", "high", "low", "close", "volume"]].to_csv(DATA / f"{t}.csv.gz")
        print(t, d.index.min().date(), d.index.max().date(), len(d))
    ff = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS")
    ff.to_csv(DATA / "FEDFUNDS.csv", index=False)
    print("FEDFUNDS", ff.iloc[0, 0], ff.iloc[-1, 0])


if __name__ == "__main__":
    main()
