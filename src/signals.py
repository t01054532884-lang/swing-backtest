"""원본 알림 규칙(analyze_frames + apply_entry_gates)을 과거 15분봉 마감마다 그대로 돌려 신호를 모은다.

속도 때문에 15분 진입 트리거(5/25 이평 교차가 최근 2봉 안, 또는 25선 위·아래로 다시 마감)가 생길 수 있는 봉만
원본 코드에 넣는다 (이평은 단순이동평균이라 창 길이와 관계없이 값이 같다).
이 트리거는 원본 규칙의 필수 조건이라, 트리거가 없는 봉은 원본 코드도 항상 WAIT를 낸다.
나머지 판단(일봉·4시간 방향, 1시간 추세, 4시간 이평 정배열, ADX, RSI, MACD, 거래량, 8개 중 점수, 손익비)은 모두 원본 코드가 한다.

python -X utf8 src/signals.py            # 모든 종목 (병렬)
결과: results/signals.csv.gz
"""

import sys
from bisect import bisect_right
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import RESULTS, TIMEFRAMES, WINDOW, candles, config, load
from astra_trader.models import D
from astra_trader.swing_strategy import SwingSettings, analyze_frames, apply_entry_gates, structure_levels


def trigger_times(m15):
    c = m15.close
    m5, m25 = c.rolling(5).mean().to_numpy(), c.rolling(25).mean().to_numpy()
    c = c.to_numpy()
    d = np.sign(m5 - m25)
    cross = np.concatenate([[False], (d[1:] != d[:-1]) & ~np.isnan(d[:-1])])
    cross = cross | np.concatenate([[False], cross[:-1]])  # 이번 봉 또는 직전 봉 교차
    r = np.sign(c - m25)
    reclaim = np.concatenate([[False], r[1:] != r[:-1]])
    return m15.close_time.to_numpy()[cross | reclaim]


def run_symbol(symbol):
    settings = SwingSettings(symbols=[symbol])  # 원본 기본값: 8개 중 6개, 손익비 1.5, 필수 게이트 켬
    frames, closes = {}, {}
    for interval in TIMEFRAMES:
        df = load(symbol, interval)
        frames[interval] = candles(df)
        closes[interval] = df.close_time.to_list()
    rows, skipped = [], {}
    for at in trigger_times(load(symbol, "15m")):
        at = int(at)
        window = {}
        for interval in TIMEFRAMES:
            end = bisect_right(closes[interval], at)
            window[interval] = frames[interval][max(0, end - WINDOW):end]
        try:
            a = apply_entry_gates(analyze_frames(window, at, settings.min_score), settings)
        except ValueError as exc:  # 상장 초기 봉 부족 등: 원본도 분석을 보류함
            key = str(exc).split(":")[0]
            skipped[key] = skipped.get(key, 0) + 1
            continue
        if a["side"] == 0:
            continue
        f = a["frames"]
        lv = {}
        if a["action"] != "WAIT":
            # 원본은 알림 순간의 호가로 계산한다. 과거에는 호가가 없어 신호 봉 종가를 쓴다.
            levels = structure_levels(a, D(f["15m"]["close"]), settings)
            lv = {"stop": float(levels["stop"]), "t1": float(levels["targets"][0]), "t2": float(levels["targets"][1]),
                  "rr": float(levels["reward_risk"]) if levels["reward_risk"] is not None else None,
                  "blocked": int(levels["blocked"] is not None), "stop_kind": "구조" if levels["support"] else "ATR"}
        rows.append({
            "symbol": symbol, "at": at, "side": a["side"], "action": a["action"], "score": a["score"],
            **{f"chk_{i}": int(v) for i, v in enumerate(a["checks"].values())},
            **{f"gate_{i}": int(v) for i, v in enumerate(a["gates"].values())},
            "close15": f["15m"]["close"], "atr1d": f["1d"]["atr14"], "atr4h": f["4h"]["atr14"],
            "vol_ratio": f["15m"]["volume_ratio"], "adx4h": f["4h"]["adx"], "rsi1h": f["1h"]["rsi14"], **lv,
        })
    print(f"{symbol}: 후보 {len(rows)}개, 점수 통과 {sum(r['action'] != 'WAIT' for r in rows)}개, 알림 {sum(r['action'] != 'WAIT' and not r.get('blocked') for r in rows)}개, 보류 {skipped}", flush=True)
    return rows


def main():
    symbols = sys.argv[1:] or config()["symbols"]
    with Pool() as pool:
        rows = [r for part in pool.map(run_symbol, symbols) for r in part]
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "signals.csv.gz", index=False)
    print(f"저장: results/signals.csv.gz ({len(rows)}행)")


if __name__ == "__main__":
    main()
