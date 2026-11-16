"""signals.csv.gz → 거래 결과, 비용·기간·민감도·무작위 비교 (results/summary.json)

python -X utf8 src/report.py
"""

import json

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, config
from simulate import Cost, Market, alerts, portfolio, random_entries, run

RULES_FIXED_MS = 1_790_640_000_000  # 2026-09-29 00:00 UTC. 최종 규칙 커밋(fb4887a) 다음 날
SEEDS = 200


def stats(t):
    if len(t) == 0:
        return {"거래": 0}
    pos, neg = t.net[t.net > 0].sum(), -t.net[t.net < 0].sum()
    return {
        "거래": len(t),
        "승률%": round(100 * (t.net > 0).mean(), 1),
        "평균R": round(t.R.mean(), 3),
        "중앙R": round(t.R.median(), 3),
        "합계R": round(t.R.sum(), 2),
        "손익비(PF)": round(pos / neg, 2) if neg else None,
        "평균 수익%": round(100 * t.net.mean(), 2),
        "평균 보유일": round(t.hold_d.mean(), 1),
        "비용R": round(((t.fees + t.funding) / t.stop_pct).mean(), 3),
    }


def book(t, name=None):
    if len(t) == 0:
        return {}
    taken, curve, locked = portfolio(t)
    if name:
        curve.to_csv(RESULTS / f"equity_{name}.csv", index=False)
    eq = np.concatenate([[10_000.0], curve.equity.to_numpy()])
    dd = 1 - eq / np.maximum.accumulate(eq)
    return {"시작": 10_000, "끝": round(eq[-1]), "수익률%": round(100 * (eq[-1] / 1e4 - 1), 2),
            "최대낙폭%": round(100 * dd.max(), 2), "들어간 거래": len(taken), "손실 잠금": locked}


def asset_group(symbol):
    for name, members in config()["groups"].items():
        if symbol in members:
            return name
    return "개별주"


def split(t):
    if len(t) == 0:
        return {}
    t = t.assign(기간=np.where(t.signal_at < RULES_FIXED_MS, "규칙 확정 전", "규칙 확정 후"),
                 자산=t.symbol.map(asset_group),
                 방향=np.where(t.side == 1, "롱", "숏"))
    out = {"전체": stats(t)}
    t = t.assign(관심목록=np.where(t.symbol.isin(config()["watchlist"]), "원래 관심 29개", "추가한 ETF·원자재 15개"))
    for col in ("기간", "자산", "관심목록", "방향", "score", "reason"):
        for k, g in t.groupby(col):
            out[f"{col}={k}"] = stats(g)
    return out


def main():
    sig = pd.read_csv(RESULTS / "signals.csv.gz")
    symbols = sorted(sig.symbol.unique())
    markets = {s: Market(s) for s in symbols}
    base = alerts(sig)
    summary = {"알림 수(손익비 통과)": len(base), "점수 통과": int((sig.action != "WAIT").sum())}
    trades = {}
    for v, name in (("B", "B 보유 알림 기준(∓10%, +10%·+20%)"), ("A", "A 진입 알림 기준(지지·저항 손절)")):
        t = run(base, markets, v)
        trades[v] = t
        t.to_csv(RESULTS / f"trades_{v}.csv", index=False)
        summary[name] = {"성과": split(t), "계좌(1% 위험, 3배 한도)": book(t, v)}
    # 비용 민감도
    summary["비용 배수(B)"] = {f"x{k}": stats(run(base, markets, "B", Cost().scaled(k))) for k in (0, 1, 2, 3)}
    summary["비용 배수(A)"] = {f"x{k}": stats(run(base, markets, "A", Cost().scaled(k))) for k in (0, 1, 2, 3)}
    # 기준 바꿔 보기 (가장 좋은 값을 고르려는 것이 아니라 결과가 기준에 얼마나 흔들리는지 보려는 것)
    summary["최소 점수(B)"] = {f"{m}/8": stats(run(alerts(sig, min_score=m), markets, "B")) for m in (6, 7, 8)}
    summary["손익비 기준(B)"] = {f"{r}": stats(run(alerts(sig, min_rr=r), markets, "B")) for r in (1.0, 1.5, 2.0, 3.0)}
    summary["손절 폭(B)"] = {f"{int(p*100)}%": stats(run(base, markets, "B", stop_pct=p)) for p in (0.05, 0.07, 0.10, 0.15)}
    # 무작위 비교: 같은 종목·같은 거래 수
    counts = trades["B"].symbol.value_counts().to_dict()
    rng = np.random.default_rng(20261116)
    real = trades["B"].R.mean()
    for follow, label in ((True, "추세 방향만 따라 아무 때나"), (False, "방향도 무작위")):
        means = []
        for _ in range(SEEDS):
            r = random_entries(markets, counts, rng, follow_trend=follow)
            means.append(r.R.mean() if len(r) else np.nan)
        means = np.array(means)
        summary[f"무작위 비교(B): {label}"] = {
            "반복": SEEDS, "무작위 평균R의 평균": round(float(np.nanmean(means)), 3),
            "무작위 5~95%": [round(float(np.nanpercentile(means, 5)), 3), round(float(np.nanpercentile(means, 95)), 3)],
            "전략 평균R": round(float(real), 3),
            "전략보다 좋은 무작위 비율%": round(100 * float(np.nanmean(means >= real)), 1),
            "무작위 평균R 목록": [round(float(x), 4) for x in means],
        }
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
