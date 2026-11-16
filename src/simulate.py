"""신호 → 거래 결과. 청산은 원본 알림 서비스가 보내는 알림을 그대로 따랐다고 가정한다.

진입: 알림이 나간 15분봉의 다음 봉 시가(+슬리피지). 그 종목에 포지션이 있으면 새 알림은 건너뜀.
청산 두 가지 (둘 다 원본 코드에 정해져 있음):
  A '진입 알림 기준'  — 진입 알림의 손절(지지·저항 아래·위, 최대 10%)과 목표가 1·2 (structure_levels)
  B '보유 알림 기준'  — 보유 감시 규칙: 평균가 ∓10% 손절, +10%·+20%에서 1/3씩 (position_levels)
공통: 목표 1·2에서 1/3씩 익절, 남은 물량은 '차트 훼손'(4시간 추세 반대, 또는 1시간·15분 추세 모두 반대)
      알림이 난 봉의 다음 시가에 정리, 보유 21일이 되면 다음 시가에 정리.
      같은 봉에서 손절과 익절이 모두 닿으면 손절을 먼저 적용. 시가가 이미 손절가를 넘었으면 시가에 체결.
비용: 수수료 편도 0.05%(테이커), 시장가 체결(진입·손절·추세·기간 청산)에 슬리피지 0.02%, 실제 펀딩비.
분할 추가매수(2·3차)는 넣지 않았다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import load, load_funding

HOLD_BARS = 21 * 96
COINS = {"BTCUSDT", "ETHUSDT", "ENAUSDT"}


@dataclass(frozen=True)
class Cost:
    fee: float = 0.0005
    slip: float = 0.0002

    def scaled(self, k):
        return Cost(self.fee * k, self.slip * k)


def frame_trend(df):
    """원본 frame_features의 trend: 종가가 60선 위이고 25선 > 60선이면 1, 반대면 -1 (단순이동평균)."""
    c = df.close
    m25, m60 = c.rolling(25).mean(), c.rolling(60).mean()
    return np.where((c > m60) & (m25 > m60), 1, np.where((c < m60) & (m25 < m60), -1, 0))


class Market:
    """한 종목의 15분봉 경로와, 각 15분봉 마감 때 볼 수 있던 상위 시간봉 추세."""

    def __init__(self, symbol):
        self.symbol = symbol
        m = load(symbol, "15m")
        self.open_time = m.open_time.to_numpy()
        self.close_time = m.close_time.to_numpy()
        self.o, self.h, self.l, self.c = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.trend = {}
        self.ready = np.ones(len(m), bool)
        for interval, need in (("15m", 201), ("1h", 201), ("4h", 201), ("1d", 61)):
            df = m if interval == "15m" else load(symbol, interval)
            t = frame_trend(df)
            idx = np.searchsorted(df.close_time.to_numpy(), self.close_time, side="right") - 1
            self.trend[interval] = np.where(idx + 1 >= need, t[np.clip(idx, 0, None)], 0)
            self.ready &= idx + 1 >= need
        f = load_funding(symbol)
        self.f_time, self.f_rate = f.time.to_numpy(), f.rate.to_numpy()
        self._broken = {s: (self.trend["4h"] == -s) | ((self.trend["1h"] == -s) & (self.trend["15m"] == -s)) for s in (1, -1)}

    def broken(self, side):
        return self._broken[side]

    def agree(self):
        d, f = self.trend["1d"], self.trend["4h"]
        return np.where(self.ready & (d == f), d, 0)

    def index_after(self, at):
        return int(np.searchsorted(self.open_time, at + 1))


def first(mask):
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else None


def trade(mk, j, side, levels, cost=Cost()):
    """j번째 봉 시가에 진입. levels(entry) → (stop, tp1, tp2). 수익은 명목 1 기준 비율."""
    n = len(mk.o)
    if j >= n:
        return None
    entry = mk.o[j] * (1 + side * cost.slip)
    stop, tp1, tp2 = levels(entry)
    if side * (entry - stop) <= 0 or side * (tp1 - entry) <= 0:
        return None  # 다음 시가가 이미 손절가 밖이거나 목표가를 넘음: 알림대로 진입할 수 없음
    end = min(n, j + HOLD_BARS + 1)
    h, l, o = mk.h[j:end], mk.l[j:end], mk.o[j:end]
    adverse, favor = (l, h) if side == 1 else (h, l)
    s = first(side * (adverse - stop) <= 0)
    t1 = first(side * (favor - tp1) >= 0)
    t2 = first(side * (favor - tp2) >= 0)
    b = first(mk.broken(side)[j:end])
    cands = []
    if s is not None:
        cands.append((s, "손절"))
    if b is not None and j + b + 1 < n:
        cands.append((b + 1, "차트 훼손"))
    if j + HOLD_BARS + 1 < n:
        cands.append((HOLD_BARS + 1, "21일"))
    k, reason = min(cands) if cands else (end - 1 - j, "보유 중")
    parts, left = [], 1.0
    for ti, price in ((t1, tp1), (t2, tp2)):
        if ti is not None and ti < k:  # 손절과 같은 봉이면 손절 먼저
            parts.append((1 / 3, price, ti))
            left -= 1 / 3
    if reason == "손절":
        px = (o[k] if side * (o[k] - stop) <= 0 else stop) * (1 - side * cost.slip)
    elif reason == "보유 중":
        px = mk.c[j + k]
    else:
        px = o[k] * (1 - side * cost.slip)
    parts.append((left, px, k))
    gross = sum(w * side * (p - entry) / entry for w, p, _ in parts)
    fees = cost.fee + sum(w * cost.fee * p / entry for w, p, _ in parts)
    t0 = mk.open_time[j]
    t_exit = mk.close_time[j + k] if reason == "보유 중" else mk.open_time[j + k]
    sel = np.flatnonzero((mk.f_time > t0) & (mk.f_time <= t_exit))
    fund = 0.0
    for i in sel:  # 펀딩: 보유 중인 비중만큼, 롱은 양수 펀딩을 낸다
        held = 1.0 - sum(w for w, _, idx in parts[:-1] if mk.open_time[j + idx] < mk.f_time[i])
        fund += side * mk.f_rate[i] * held
    net = gross - fees - fund
    risk = side * (entry - stop) / entry
    return {
        "symbol": mk.symbol, "side": side, "entry_time": int(t0), "exit_time": int(t_exit), "entry": entry,
        "stop_pct": risk, "reason": reason, "tp1": len(parts) > 1, "tp2": len(parts) > 2, "hold_d": k / 96,
        "gross": gross, "fees": fees, "funding": fund, "net": net, "R": net / risk, "exit_index": j + k,
    }


def levels_alert(r):
    return lambda entry: (r.stop, r.t1, r.t2)


def levels_hold(side, pct=0.10):
    return lambda entry: (entry * (1 - side * pct), entry * (1 + side * pct), entry * (1 + side * 2 * pct))


def alerts(signals, min_score=6, min_rr=1.5):
    """원본이 실제로 알림을 보냈을 신호: 점수·필수 게이트 통과 + 손익비 기준."""
    s = signals[(signals.action != "WAIT") & (signals.score >= min_score)]
    return s[s.rr.isna() | (s.rr >= min_rr)]


def run(signals, markets, variant="B", cost=Cost(), stop_pct=0.10):
    out = []
    for symbol, g in signals.groupby("symbol"):
        mk = markets[symbol]
        free_from = -1
        for r in g.sort_values("at").itertuples():
            j = mk.index_after(r.at)
            if j <= free_from:
                continue
            lv = levels_alert(r) if variant == "A" else levels_hold(r.side, stop_pct)
            res = trade(mk, j, r.side, lv, cost)
            if res is None:
                continue
            res.update(score=r.score, signal_at=r.at)
            out.append(res)
            free_from = res["exit_index"]
    return pd.DataFrame(out)


def random_entries(markets, counts, rng, cost=Cost(), stop_pct=0.10, follow_trend=True):
    """비교용: 같은 종목·같은 거래 수로 아무 때나 진입 (추세 방향만 따르거나, 방향도 무작위)."""
    out = []
    for symbol, n in counts.items():
        mk = markets[symbol]
        side_at = mk.agree() if follow_trend else np.where(mk.ready, rng.choice([-1, 1], len(mk.o)), 0)
        idx = np.flatnonzero(side_at != 0)
        idx = idx[idx + 1 < len(mk.o)]
        if len(idx) == 0:
            continue
        free_from = -1
        for i in np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)):
            if i + 1 <= free_from:
                continue
            side = int(side_at[i])
            res = trade(mk, i + 1, side, levels_hold(side, stop_pct), cost)
            if res:
                out.append(res)
                free_from = res["exit_index"]
    return pd.DataFrame(out)


def portfolio(trades, start=10_000.0, risk=0.01, leverage=3, fee_buffer=0.002, loss_lock=0.20):
    """원본 수량 규칙: 손절 시 손실이 순자산 1%가 되게, 전체 명목은 순자산 3배까지, 손실 20%면 신규 진입 잠금."""
    events = sorted([(t.entry_time, 1, i) for i, t in trades.iterrows()] + [(t.exit_time, 0, i) for i, t in trades.iterrows()])
    equity, open_pos, taken, curve, locked = start, {}, {}, [], False
    for time, kind, i in events:
        t = trades.loc[i]
        if kind == 1:
            if locked:
                continue
            notional = min(risk * equity / (t.stop_pct + fee_buffer), max(0.0, leverage * equity - sum(open_pos.values())))
            if notional > 0:
                open_pos[i] = notional
        elif i in open_pos:
            notional = open_pos.pop(i)
            equity += notional * t.net
            taken[i] = notional
            curve.append((time, equity))
            locked = locked or equity <= start * (1 - loss_lock)
    out = trades.loc[list(taken)].copy()
    out["notional"] = list(taken.values())
    out["pnl"] = out.notional * out.net
    return out, pd.DataFrame(curve, columns=["time", "equity"]), locked
