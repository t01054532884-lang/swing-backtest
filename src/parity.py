"""청산에 쓰는 빠른 추세 계산(simulate.Market)이 원본 frame_features의 trend와 같은지 무작위 시점에서 대조.

python -X utf8 src/parity.py BTCUSDT NVDAUSDT
"""

import sys
from bisect import bisect_right

import numpy as np

from common import TIMEFRAMES, WINDOW, candles, load
from simulate import Market
from astra_trader.swing_strategy import frame_features


def main():
    rng = np.random.default_rng(7)
    total = same = 0
    for symbol in sys.argv[1:] or ["BTCUSDT"]:
        mk = Market(symbol)
        frames = {i: load(symbol, i) for i in TIMEFRAMES}
        objs = {i: candles(df) for i, df in frames.items()}
        for k in rng.choice(np.flatnonzero(mk.ready), 300, replace=False):
            at = int(mk.close_time[k])
            for i in TIMEFRAMES:
                end = bisect_right(frames[i].close_time.to_list(), at)
                f = frame_features(objs[i][max(0, end - WINDOW):end], daily=i == "1d")
                total += 1
                same += int(f["trend"] == mk.trend[i][k])
        print(f"{symbol}: 누적 {same}/{total} 일치")
    print(f"일치율 {100 * same / total:.2f}%")


if __name__ == "__main__":
    main()
