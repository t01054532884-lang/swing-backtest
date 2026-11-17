"""ETF 추세추종 백테스트. 설계는 etf/DESIGN.md (결과 보기 전에 고정).

python -X utf8 etf/backtest_etf.py   → results/etf/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = ROOT / "results" / "etf"
RISK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "IEF"]
CASH = "SHY"
LOOKBACKS = [3, 6, 9, 12]
START = "2007-02-28"   # DBC 상장(2006-02) 뒤 12개월 수익률을 처음 잴 수 있는 월말
TRAIN_END = "2015-12-31"
TEST_START = "2016-01-01"
COST = 0.0010


def load():
    o, c = {}, {}
    for t in RISK + [CASH]:
        d = pd.read_csv(HERE / "data" / f"{t}.csv.gz", index_col="date", parse_dates=True)
        o[t], c[t] = d.open, d.close
    o, c = pd.DataFrame(o), pd.DataFrame(c)
    return o, c


def month_ends(close):
    return close.groupby(close.index.to_period("M")).apply(lambda d: d.index[-1]).tolist()


def targets(close, me, rule, L):
    """월말 me[i] 종가로 정한 목표 비중 (다음 거래일 시가에 체결)."""
    out = {}
    for i, d in enumerate(me):
        if d < pd.Timestamp(START) or i < L:
            continue
        past = me[i - L]
        ret = close.loc[d] / close.loc[past] - 1
        if ret[RISK + [CASH]].isna().any():
            continue
        trend = ret[RISK] > ret[CASH]
        w = pd.Series(0.0, index=RISK + [CASH])
        if rule == "S1":
            for t in RISK:
                w[t if trend[t] else CASH] += 1 / len(RISK)
        elif rule == "S2":
            for t in ret[RISK].sort_values(ascending=False).index[:3]:
                w[t if trend[t] else CASH] += 1 / 3
        elif rule == "EW":
            w[RISK] = 1 / len(RISK)
        elif rule == "6040":
            w["SPY"], w["IEF"] = 0.6, 0.4
        elif rule == "SPY":
            w["SPY"] = 1.0
        out[d] = w
    return out


def simulate(opn, close, tw, cost=COST, rebalance=True):
    """보유 수량 기준 일별 평가. 목표 비중은 판단일 다음 거래일 시가에 맞춘다."""
    days = close.index
    exec_on = {}
    for d, w in tw.items():
        nxt = days[days > d]
        if len(nxt):
            exec_on[nxt[0]] = w
    first = min(exec_on)
    shares = pd.Series(0.0, index=close.columns)
    value, rows, turnover = 1.0, [], []
    started = False
    for day in days[days >= first]:
        if day in exec_on and (rebalance or not started):
            px = opn.loc[day]
            v_open = float((shares * px).sum()) if started else value
            cur = (shares * px) / v_open if started else pd.Series(0.0, index=close.columns)
            w = exec_on[day].reindex(close.columns).fillna(0.0)
            to = float((w - cur.fillna(0)).abs().sum())
            v_after = v_open * (1 - cost * to)
            shares = (w * v_after / px).fillna(0.0)
            turnover.append((day, to))
            started = True
        value = float((shares * close.loc[day]).sum())
        rows.append((day, value))
    eq = pd.Series(dict(rows))
    return eq, pd.Series(dict(turnover))


def metrics(eq, rf, to=None):
    r = eq.pct_change().dropna()
    ex = r - rf.reindex(r.index).fillna(0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = 1 - eq / eq.cummax()
    yearly = eq.resample("YE").last().pct_change().dropna()
    monthly = eq.resample("ME").last().pct_change().dropna()
    m = {
        "연수익률%": 100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1),
        "연변동성%": 100 * r.std() * np.sqrt(252),
        "샤프": ex.mean() / ex.std() * np.sqrt(252),
        "최대낙폭%": 100 * dd.max(),
        "최악의 해%": 100 * yearly.min() if len(yearly) else np.nan,
        "월 양수%": 100 * (monthly > 0).mean(),
    }
    if to is not None and len(to):
        m["연 회전율"] = to[(to.index >= eq.index[0]) & (to.index <= eq.index[-1])].sum() / years
    return {k: round(float(v), 2) for k, v in m.items()}


def block_bootstrap_sharpe_diff(a, b, rf, n=2000, block=12, seed=7):
    """월 초과수익으로 샤프 차이(a − b)의 90% 구간. 12개월 블록을 원형으로 이어 붙임."""
    ma = a.resample("ME").last().pct_change().dropna()
    mb = b.resample("ME").last().pct_change().dropna()
    mr = (1 + rf).resample("ME").prod() - 1
    idx = ma.index.intersection(mb.index)
    xa, xb = (ma - mr.reindex(ma.index)).loc[idx].to_numpy(), (mb - mr.reindex(mb.index)).loc[idx].to_numpy()
    T = len(idx)
    rng = np.random.default_rng(seed)
    sh = lambda x: x.mean() / x.std() * np.sqrt(12)
    diffs = []
    for _ in range(n):
        starts = rng.integers(0, T, size=int(np.ceil(T / block)))
        sel = np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
        diffs.append(sh(xa[sel]) - sh(xb[sel]))
    diffs = np.array(diffs)
    return {"관측": round(float(sh(xa) - sh(xb)), 3), "90% 구간": [round(float(np.percentile(diffs, 5)), 3), round(float(np.percentile(diffs, 95)), 3)],
            "0보다 클 비율%": round(float(100 * (diffs > 0).mean()), 1), "개월": T}


def rate_regimes(close):
    ff = pd.read_csv(HERE / "data" / "FEDFUNDS.csv", parse_dates=["observation_date"]).set_index("observation_date").FEDFUNDS
    ff.index = ff.index.to_period("M")
    mret = close.resample("ME").last().pct_change()
    mret.index = mret.index.to_period("M")
    rows = []
    for m in mret.index:
        # m월 수익률을 m-1월 말에 알 수 있던 정보로 나눈다: 금리는 m-2월 값과 m-8월 값의 차이 (발표 시차 고려)
        a, b = m - 2, m - 8
        if a in ff.index and b in ff.index:
            ch = ff[a] - ff[b]
            rows.append((m, "인하" if ch <= -0.25 else "인상" if ch >= 0.25 else "유지"))
    reg = pd.Series(dict(rows))
    out = {}
    for name in ("인하", "유지", "인상"):
        ms = reg[reg == name].index.intersection(mret.index)
        sub = mret.loc[ms]
        out[name] = {"개월": int(len(ms)), **{t: round(float(100 * sub[t].mean()), 2) for t in RISK + [CASH]}}
    cut = reg[reg == "인하"].index.intersection(mret.index)
    gaps = pd.Series(cut.to_timestamp()).diff().dt.days.fillna(999) > 40
    out["인하 구간 묶음 수"] = int(gaps.sum())
    for a in ("GLD", "DBC", "TLT"):
        d = (mret.loc[cut, a] - mret.loc[cut, "SPY"]).dropna()
        out[f"인하 구간 {a}−SPY"] = {"월평균 차이%p": round(float(100 * d.mean()), 2),
                                   "t값": round(float(d.mean() / d.std() * np.sqrt(len(d))), 2), "개월": int(len(d))}
    return out, reg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opn, close = load()
    me = month_ends(close)
    rf = close[CASH].pct_change()
    train = slice(START, TRAIN_END)
    test = slice(TEST_START, None)
    summary = {"설계": "etf/DESIGN.md", "비용(편도)": COST}

    # 1) 학습 기간에서 L 고르기
    grid, curves = {}, {}
    for rule in ("S1", "S2"):
        for L in LOOKBACKS:
            for c in (COST, 0.0020, 0.0030):
                eq, to = simulate(opn, close, targets(close, me, rule, L), cost=c)
                curves[(rule, L, c)] = (eq, to)
            eq, to = curves[(rule, L, COST)]
            grid[f"{rule} L={L}"] = {"학습": metrics(eq[train] / eq[train].iloc[0], rf, to), "검증": metrics(eq[test] / eq[test].iloc[0], rf, to)}
    chosen = {r: max(LOOKBACKS, key=lambda L: grid[f"{r} L={L}"]["학습"]["샤프"]) for r in ("S1", "S2")}
    summary["학습에서 고른 L(개월)"] = chosen
    summary["L별 결과(흔들림 확인)"] = grid

    # 2) 비교 대상
    bench = {}
    for name, rule in (("SPY 보유", "SPY"), ("60/40", "6040"), ("10개 동일 비중", "EW")):
        eq, to = simulate(opn, close, targets(close, me, rule, 12), rebalance=rule != "SPY")
        bench[name] = (eq, to)

    # 3) 검증 기간 본 결과 (학습에서 고른 L 그대로)
    main_set = {f"S1 추세 필터 (L={chosen['S1']})": curves[("S1", chosen["S1"], COST)],
                f"S2 상대 강도 상위 3 (L={chosen['S2']})": curves[("S2", chosen["S2"], COST)], **bench}
    table = {}
    for period, sl in (("학습 2007-03~2015", train), ("검증 2016~2026-11", test), ("전체", slice(START, None))):
        table[period] = {k: metrics(eq[sl] / eq[sl].iloc[0], rf, to) for k, (eq, to) in main_set.items()}
    summary["결과"] = table
    summary["비용 스트레스(검증)"] = {
        f"{r} L={chosen[r]}": {f"{c*100:.1f}%": metrics(curves[(r, chosen[r], c)][0][test] / curves[(r, chosen[r], c)][0][test].iloc[0], rf)["샤프"] for c in (COST, 0.0020, 0.0030)}
        for r in ("S1", "S2")}
    s1 = curves[("S1", chosen["S1"], COST)][0][test]
    s2 = curves[("S2", chosen["S2"], COST)][0][test]
    summary["샤프 차이 부트스트랩(검증)"] = {
        "S1 − 10개 동일 비중": block_bootstrap_sharpe_diff(s1, bench["10개 동일 비중"][0][test], rf),
        "S1 − SPY": block_bootstrap_sharpe_diff(s1, bench["SPY 보유"][0][test], rf),
        "S2 − SPY": block_bootstrap_sharpe_diff(s2, bench["SPY 보유"][0][test], rf),
    }
    # 4) 같은 변동성으로 맞춘 비교: 동일 비중 + SHY를 섞어 S1과 변동성을 같게 (매일 재조정, 비용 무시 = 비교 대상에 유리)
    vm = {}
    for period, sl in (("학습", train), ("검증", test)):
        s1r = curves[("S1", chosen["S1"], COST)][0][sl].pct_change().dropna()
        ewr = bench["10개 동일 비중"][0][sl].pct_change().dropna()
        cr = rf.reindex(ewr.index).fillna(0)
        k = float(s1r.std() / ewr.std())
        mix = (1 + k * ewr + (1 - k) * cr).cumprod()
        vm[period] = {"동일 비중 비율": round(k, 2), "혼합": metrics(mix, rf), "S1": metrics(curves[("S1", chosen["S1"], COST)][0][sl] / curves[("S1", chosen["S1"], COST)][0][sl].iloc[0], rf)}
    summary["변동성 맞춘 비교(S1 vs 동일 비중+SHY)"] = vm
    reg, regs = rate_regimes(close)
    summary["금리 구간별 다음 달 평균 수익률%(H3)"] = reg

    # 저장
    eqs = pd.DataFrame({k: eq for k, (eq, _) in main_set.items()})
    eqs.index.name = "date"
    eqs.to_csv(OUT / "equity.csv")
    w_last = targets(close, me, "S1", chosen["S1"])
    last_d = max(w_last)
    summary["마지막 판단(S1)"] = {"날짜": str(last_d.date()) + " (데이터 마지막 날 기준)", "비중": {k: round(float(v), 3) for k, v in w_last[last_d].items() if v > 0}}
    regs.astype(str).to_csv(OUT / "rate_regime.csv", header=["구간"])
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("학습에서 고른 L(개월)", "결과", "비용 스트레스(검증)", "샤프 차이 부트스트랩(검증)")}, ensure_ascii=False, indent=1))
    print(json.dumps(summary["금리 구간별 다음 달 평균 수익률%(H3)"], ensure_ascii=False))


if __name__ == "__main__":
    main()
