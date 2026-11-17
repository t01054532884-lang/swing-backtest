"""스윙 알림 규칙 검증 결과 웹 화면.  streamlit run app.py"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
RES = ROOT / "results"
RULES_FIXED = pd.Timestamp("2026-09-29", tz="UTC")
BLUE, ORANGE, GRAY, GREEN, RED = "#2F5BD3", "#C2410C", "#94A3B8", "#15803D", "#B91C1C"
VARIANTS = {
    "A": ("A · 진입 알림 기준", "진입 알림에 적힌 손절(가까운 지지·저항 너머, 최대 10%)과 목표가 1·2"),
    "B": ("B · 보유 알림 기준", "보유 감시 규칙: 평균가 대비 10% 손절, +10%·+20%에서 1/3씩 익절"),
}

st.set_page_config(page_title="투자 규칙 검증", page_icon="📈", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .note {color:#64748B; font-size:0.9rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    trades, equity = {}, {}
    for v in VARIANTS:
        t = pd.read_csv(RES / f"trades_{v}.csv")
        for c in ("entry_time", "exit_time", "signal_at"):
            t[c] = pd.to_datetime(t[c], unit="ms", utc=True)
        trades[v] = t
        e = pd.read_csv(RES / f"equity_{v}.csv")
        e["time"] = pd.to_datetime(e.time, unit="ms", utc=True)
        equity[v] = e
    return cfg, summary, trades, equity


@st.cache_data
def bars():
    b = pd.read_csv(RES / "bars_1h.csv.gz")
    b["time"] = pd.to_datetime(b.open_time, unit="ms", utc=True)
    return b


cfg, summary, trades, equity = load()


def group(symbol):
    for name, members in cfg["groups"].items():
        if symbol in members:
            return name
    return "개별주"


def stats(t):
    if len(t) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, pf=np.nan, total=0.0)
    pos, neg = t.R[t.R > 0].sum(), -t.R[t.R < 0].sum()
    return dict(n=len(t), win=100 * (t.net > 0).mean(), avg=t.R.mean(), pf=pos / neg if neg else np.nan, total=t.R.sum())


# ── 머리말 ───────────────────────────────────────────────
with st.sidebar:
    study = st.radio("연구", ["① 스윙 알림 규칙", "② 미국 ETF 추세추종"])
    st.divider()

st.title("투자 규칙, 비용까지 넣어 다시 검증")
st.markdown(
    "① 직접 만들어 Discord로 받아 쓰던 **바이낸스 선물 스윙 알림** 규칙을 실제 코드 그대로 과거 시세에 돌렸고, "
    "비용을 넣으면 남지 않는다는 결론이 나와 ② 데이터가 길고 경제적 이유가 있는 **미국 ETF 추세추종**으로 바꿔 다시 검증했습니다. "
    "두 연구 모두 결과를 보고 기준을 다시 맞추지 않았습니다. 투자 권유가 아닙니다."
)


def etf_page():
    e = json.loads((RES / "etf" / "summary.json").read_text(encoding="utf-8"))
    eq = pd.read_csv(RES / "etf" / "equity.csv", index_col="date", parse_dates=True)
    st.header("② 미국 ETF 추세추종")
    st.markdown(
        "**가설** 추세가 꺾인 자산을 단기채(SHY)로 바꿔 두면, 그냥 들고 있을 때보다 낙폭이 작고 위험 대비 성과가 낫다.  " + "\n"
        "**방법** ETF 10개(SPY QQQ IWM EFA EEM VNQ GLD DBC TLT IEF) · 매달 말 판단, 다음 날 시가 체결 · 비용 0.1% · "
        "2007~2015에서 기준(추세 기간)을 고르고 **2016~2026은 한 번만 계산** · 설계는 결과 전에 고정(etf/DESIGN.md)"
    )
    test = e["결과"]["검증 2016~2026-11"]
    names = list(test)
    c = st.columns(4)
    s1 = test[names[0]]
    ew = test["10개 동일 비중"]
    c[0].metric("S1 최대 낙폭 (검증)", f"{s1['최대낙폭%']:.1f}%", f"동일 비중 {ew['최대낙폭%']:.1f}%", delta_color="off")
    c[1].metric("S1 샤프 (검증)", f"{s1['샤프']:.2f}", f"동일 비중 {ew['샤프']:.2f}", delta_color="off")
    c[2].metric("S1 연수익률 (검증)", f"{s1['연수익률%']:.1f}%", f"동일 비중 {ew['연수익률%']:.1f}%", delta_color="off")
    c[3].metric("학습에서 고른 추세 기간", f"{e['학습에서 고른 L(개월)']['S1']}개월")
    period = st.radio("기간", ["전체", "학습 2007~2015", "검증 2016~2026"], horizontal=True, index=0)
    sl = {"전체": slice(None), "학습 2007~2015": slice(None, "2015-12-31"), "검증 2016~2026": slice("2016-01-01", None)}[period]
    q = eq.loc[sl]
    q = q / q.iloc[0]
    colors = [BLUE, "#7C3AED", ORANGE, GREEN, GRAY]
    fig = go.Figure()
    for (name, col), color in zip(q.items(), colors):
        fig.add_trace(go.Scatter(x=q.index, y=col, name=name, line=dict(color=color, width=2.4 if name.startswith("S1") else 1.3)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(type="log", title="누적 (시작 = 1)", tickvals=[0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8], ticktext=["0.6", "0.8", "1", "1.5", "2", "3", "4", "6", "8"]), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    for (name, col), color in zip(q.items(), colors):
        fig.add_trace(go.Scatter(x=q.index, y=-100 * (1 - col / col.cummax()), name=name, line=dict(color=color, width=2 if name.startswith("S1") else 1), showlegend=False))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="고점 대비 %")
    st.plotly_chart(fig, use_container_width=True)
    rows = []
    for p_name, block in e["결과"].items():
        for k, m in block.items():
            rows.append({"기간": p_name, "전략": k, "연수익률%": m["연수익률%"], "연변동성%": m["연변동성%"], "샤프": m["샤프"], "최대낙폭%": m["최대낙폭%"], "최악의 해%": m["최악의 해%"], "연 회전율": m.get("연 회전율")})
    st.subheader("숫자로 보기")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    vm = e["변동성 맞춘 비교(S1 vs 동일 비중+SHY)"]
    st.subheader("그냥 현금을 섞은 것보다 나은가? (변동성을 같게 맞춰 비교)")
    st.dataframe(pd.DataFrame([{"기간": k, "S1 연수익률%": v["S1"]["연수익률%"], "혼합 연수익률%": v["혼합"]["연수익률%"], "S1 최대낙폭%": v["S1"]["최대낙폭%"], "혼합 최대낙폭%": v["혼합"]["최대낙폭%"]} for k, v in vm.items()]), hide_index=True, use_container_width=True)
    st.caption("혼합 = 10개 동일 비중과 SHY를 섞어 S1과 변동성을 같게 만든 것 (매일 재조정, 비용 없음 — 혼합 쪽에 유리한 가정)")
    st.subheader("흔들림 확인")
    grid = e["L별 결과(흔들림 확인)"]
    st.dataframe(pd.DataFrame([{"전략·추세 기간": k, "학습 샤프": v["학습"]["샤프"], "검증 샤프": v["검증"]["샤프"], "검증 최대낙폭%": v["검증"]["최대낙폭%"]} for k, v in grid.items()]), hide_index=True, use_container_width=True)
    bs = e["샤프 차이 부트스트랩(검증)"]
    st.caption("검증 기간 샤프 차이 (12개월 블록 부트스트랩 2,000번, 90% 구간): " + " · ".join(f"{k} {v['관측']:+.2f} [{v['90% 구간'][0]:+.2f}, {v['90% 구간'][1]:+.2f}]" for k, v in bs.items()))
    h3 = e["금리 구간별 다음 달 평균 수익률%(H3)"]
    st.subheader("금리 인하 구간에는 금·원자재가 나을까? (모의투자 때 생각)")
    st.dataframe(pd.DataFrame([{"구간": k, "개월": v["개월"], **{t: v[t] for t in ("SPY", "QQQ", "GLD", "DBC", "TLT", "SHY")}} for k, v in h3.items() if k in ("인하", "유지", "인상")]), hide_index=True, use_container_width=True)
    g = h3["인하 구간 GLD−SPY"]
    st.caption(f"월평균 수익률(%). 인하 구간에서 금은 SPY보다 월 {g['월평균 차이%p']:+.2f}%p 높았지만 t값 {g['t값']}, 인하 국면이 {h3['인하 구간 묶음 수']}번뿐이라 통계적으로 확인됐다고 말할 수 없습니다.")
    st.subheader("읽은 것")
    st.markdown(
        "1. 추세 필터는 두 기간 모두 **최대 낙폭을 크게 줄였습니다** (변동성을 맞춰도 학습 22.8→11.0%, 검증 16.9→12.6%)." + "\n"
        "2. 하지만 **위험 대비 성과는 2008년이 있던 학습 기간에서만** 나았고, 검증 기간에는 동일 비중·SPY보다 낮았습니다. 큰 하락장을 피하는 보험에 가깝습니다." + "\n"
        "3. 학습에서 고른 12개월이 검증에서는 가장 나빴습니다(검증 샤프 0.51, 다른 기간은 0.62~0.63). 고른 값 하나보다 범위로 판단해야 합니다." + "\n"
        "4. 자세한 방법과 한계: github.com/t01054532884-lang/swing-backtest/tree/main/etf"
    )


if study.startswith("②"):
    etf_page()
    st.stop()

st.header("① 스윙 알림 규칙")
st.markdown(
    "직접 만들어 Discord로 받아 쓰던 **바이낸스 선물 스윙 알림**의 규칙을, 실제 알림 코드 그대로 과거 시세에 돌렸습니다. "
    "규칙이 낸 신호는 **골라내지 않고 전부** 기록했고, 손절로 끝난 거래도 그대로 남겼습니다."
)
st.markdown(
    f"<span class='note'>종목 {len(cfg['symbols'])}개 (원래 관심 목록 {len(cfg['watchlist'])}개 + ETF·원자재 {len(cfg['added_etf'])}개) · "
    "2022-01 ~ 2026-11-15 · 수수료·슬리피지·실제 펀딩비 포함 · 규칙 확정 2026-09-29 이후로 규칙을 바꾸지 않음</span>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("보기 설정")
    v = st.radio("청산 방식", list(VARIANTS), format_func=lambda k: VARIANTS[k][0])
    st.caption(VARIANTS[v][1])
    groups = sorted({group(s) for s in trades[v].symbol})
    picked = st.multiselect("자산", groups, default=groups)
    period = st.radio("기간", ["전체", "규칙 확정 전", "규칙 확정 후 (처음 보는 구간)"])
    side = st.radio("방향", ["전체", "롱", "숏"], horizontal=True)
    st.divider()
    st.caption("코드: github.com/t01054532884-lang/swing-backtest")

t = trades[v].copy()
t["자산"] = t.symbol.map(group)
t = t[t["자산"].isin(picked)]
if period != "전체":
    after = t.signal_at >= RULES_FIXED
    t = t[after] if period.startswith("규칙 확정 후") else t[~after]
if side != "전체":
    t = t[t.side == (1 if side == "롱" else -1)]
s = stats(t)

c = st.columns(5)
c[0].metric("거래", f"{s['n']:,}건")
c[1].metric("승률", f"{s['win']:.1f}%" if s["n"] else "-")
c[2].metric("평균 R", f"{s['avg']:+.3f}" if s["n"] else "-", help="R = 손절 한 번의 손실을 1로 본 손익. 비용 포함")
c[3].metric("손익비 (PF)", f"{s['pf']:.2f}" if s["n"] and not np.isnan(s["pf"]) else "-", help="이익 R 합 ÷ 손실 R 합. 1보다 커야 남음")
c[4].metric("합계 R", f"{s['total']:+.1f}" if s["n"] else "-")

tab1, tab2, tab3, tab4 = st.tabs(["한눈에 보기", "거래 기록", "차트로 보기", "방법과 한계"])

# ── 한눈에 보기 ───────────────────────────────────────────
with tab1:
    st.subheader("가상 계좌 1만 USDT (거래당 순자산 1% 위험, 명목 3배 한도)")
    fig = go.Figure()
    for k, color in (("A", BLUE), ("B", ORANGE)):
        e = equity[k]
        fig.add_trace(go.Scatter(x=e.time, y=e.equity, name=VARIANTS[k][0], line=dict(color=color, width=2 if k == v else 1),
                                 opacity=1 if k == v else 0.45))
    fig.add_hline(y=10_000, line=dict(color=GRAY, width=1))
    fig.add_vline(x=RULES_FIXED, line=dict(color="#475569", dash="dash", width=1))
    fig.add_annotation(x=RULES_FIXED, y=1, yref="paper", text="규칙 확정", showarrow=False, xanchor="left", font=dict(size=11))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08), yaxis_title="USDT")
    st.plotly_chart(fig, use_container_width=True)
    book = summary[[k for k in summary if k.startswith(v)][0]]["계좌(1% 위험, 3배 한도)"]
    st.caption(f"{VARIANTS[v][0]}: 1만 → {book['끝']:,} ({book['수익률%']:+.1f}%), 최대 낙폭 {book['최대낙폭%']:.1f}%, 들어간 거래 {book['들어간 거래']}건 · 계좌 곡선은 전체 종목 기준 (왼쪽 필터와 무관)")

    left, right = st.columns(2)
    with left:
        st.subheader("자산별 평균 R")
        g = trades[v].assign(자산=trades[v].symbol.map(group)).groupby("자산").agg(거래=("R", "size"), 평균R=("R", "mean")).reset_index()
        fig = go.Figure(go.Bar(x=g["자산"], y=g["평균R"], marker_color=[GREEN if x > 0 else RED for x in g["평균R"]],
                               text=[f"{x:+.3f}<br>{n}건" for x, n in zip(g["평균R"], g["거래"])], textposition="outside"))
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="평균 R")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("비용을 늘리면")
        cost = summary[f"비용 배수({v})"]
        xs = list(cost)
        fig = go.Figure(go.Scatter(x=[x.replace("x", "×") for x in xs], y=[cost[x]["평균R"] for x in xs], mode="lines+markers+text",
                                   text=[f"{cost[x]['평균R']:+.3f}" for x in xs], textposition="top center", line=dict(color=BLUE if v == "A" else ORANGE)))
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="수수료·슬리피지 배수", yaxis_title="평균 R")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("아무 때나 들어갔다면? (B 기준, 200번)")
    rnd = summary["무작위 비교(B): 추세 방향만 따라 아무 때나"]
    fig = go.Figure(go.Histogram(x=rnd["무작위 평균R 목록"], nbinsx=30, marker_color=GRAY, name="무작위"))
    fig.add_vline(x=rnd["전략 평균R"], line=dict(color=ORANGE, width=3))
    fig.add_annotation(x=rnd["전략 평균R"], y=1, yref="paper", text="이 규칙", showarrow=False, xanchor="right", font=dict(color=ORANGE))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="평균 R", yaxis_title="횟수", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"같은 종목에서 같은 수만큼, 일봉·4시간 추세 방향으로 아무 때나 들어가 같은 규칙으로 청산. 200번 중 {rnd['전략보다 좋은 무작위 비율%']}%가 이 규칙보다 좋았습니다.")

# ── 거래 기록 ─────────────────────────────────────────────
with tab2:
    st.markdown("규칙이 낸 신호를 **전부** 시간순으로 보여 줍니다. 결과를 보고 고른 거래가 아닙니다.")
    view = t.sort_values("entry_time", ascending=False).assign(
        진입=lambda d: d.entry_time.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M"),
        방향=lambda d: np.where(d.side == 1, "롱", "숏"),
        진입가=lambda d: d.entry.round(4), 손절가=lambda d: d.stop.round(4),
        목표1=lambda d: d.tp1_px.round(4), 목표2=lambda d: d.tp2_px.round(4),
        결과=lambda d: d.reason + np.where(d.tp2, " · 목표2", np.where(d.tp1, " · 목표1", "")),
        수익률=lambda d: (100 * d.net).round(2), R=lambda d: d.R.round(2), 보유일=lambda d: d.hold_d.round(1),
    )[["진입", "symbol", "자산", "방향", "score", "진입가", "손절가", "목표1", "목표2", "결과", "보유일", "수익률", "R"]]
    view = view.rename(columns={"symbol": "종목", "score": "점수(8개 중)", "수익률": "수익률%"})
    st.dataframe(view, use_container_width=True, hide_index=True, height=520,
                 column_config={"R": st.column_config.NumberColumn(format="%+.2f"), "수익률%": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("시각은 한국 시간. 수익률은 명목 금액 대비, 비용 포함. 목표1·2에서 1/3씩 익절하고 남은 물량은 '차트 훼손' 알림이나 21일에 정리.")

# ── 차트로 보기 ───────────────────────────────────────────
with tab3:
    if len(t) == 0:
        st.info("왼쪽 조건에 맞는 거래가 없습니다.")
    else:
        sym = st.selectbox("종목", sorted(t.symbol.unique()), index=sorted(t.symbol.unique()).index("BTCUSDT") if "BTCUSDT" in set(t.symbol) else 0)
        ts = t[t.symbol == sym].sort_values("entry_time", ascending=False).reset_index(drop=True)
        labels = [f"{r.entry_time.tz_convert('Asia/Seoul'):%Y-%m-%d %H:%M} · {'롱' if r.side == 1 else '숏'} · {r.reason} · {r.R:+.2f}R" for r in ts.itertuples()]
        i = st.selectbox("거래", range(len(ts)), format_func=lambda k: labels[k])
        r = ts.loc[i]
        b = bars()
        b = b[(b.symbol == sym) & (b.time >= r.entry_time - pd.Timedelta(days=4)) & (b.time <= r.exit_time + pd.Timedelta(days=3))]
        fig = go.Figure(go.Candlestick(x=b.time, open=b.open, high=b.high, low=b.low, close=b.close, name=sym,
                                       increasing_line_color=GREEN, decreasing_line_color=RED))
        x0, x1 = r.entry_time, r.exit_time
        for y, name, color, dash in ((r.entry, "진입", "#0F172A", "solid"), (r.stop, "손절", RED, "dash"),
                                     (r.tp1_px, "목표1", GREEN, "dot"), (r.tp2_px, "목표2", GREEN, "dash")):
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=color, dash=dash, width=1.5))
            fig.add_annotation(x=x1, y=y, text=f"{name} {y:,.2f}" if y >= 10 else f"{name} {y:.4f}", showarrow=False, xanchor="left", font=dict(color=color, size=11))
        fig.add_trace(go.Scatter(x=[x0], y=[r.entry], mode="markers", marker=dict(symbol="triangle-up" if r.side == 1 else "triangle-down", size=14, color=BLUE), name="진입"))
        fig.add_trace(go.Scatter(x=[x1], y=[r.exit_px], mode="markers", marker=dict(symbol="x", size=12, color=ORANGE), name=f"정리 ({r.reason})"))
        fig.update_layout(height=520, margin=dict(l=10, r=80, t=10, b=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, use_container_width=True)
        cc = st.columns(4)
        cc[0].metric("결과", r.reason)
        cc[1].metric("수익률 (비용 포함)", f"{100 * r.net:+.2f}%")
        cc[2].metric("R", f"{r.R:+.2f}")
        cc[3].metric("보유", f"{r.hold_d:.1f}일")
        st.caption("1시간봉으로 그렸습니다. 판정·체결은 15분봉 기준이라 그림과 아주 조금 다를 수 있습니다.")

# ── 방법과 한계 ───────────────────────────────────────────
with tab4:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.find("### 가설")
    st.markdown(readme[start:].replace("![계좌 곡선](docs/equity.png)", ""))
