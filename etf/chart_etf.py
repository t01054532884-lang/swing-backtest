"""docs/etf_equity.png: 누적 수익(로그)과 고점 대비 하락"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parents[1]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

eq = pd.read_csv(ROOT / "results" / "etf" / "equity.csv", index_col="date", parse_dates=True)
eq = eq / eq.iloc[0]
colors = ["#2F5BD3", "#7C3AED", "#C2410C", "#15803D", "#94A3B8"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for (name, s), c in zip(eq.items(), colors):
    lw = 2.0 if name.startswith("S1") else 1.3
    a1.plot(s.index, s, label=name, color=c, lw=lw)
    a2.plot(s.index, -100 * (1 - s / s.cummax()), color=c, lw=lw)
for a in (a1, a2):
    a.axvspan(pd.Timestamp("2016-01-01"), s.index[-1], color="#EEF1F6", zorder=0)
    a.spines[["top", "right"]].set_visible(False)
a1.set_yscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
a1.yaxis.set_major_locator(FixedLocator([0.6, 1, 1.5, 2, 3, 4, 6, 8]))
a1.yaxis.set_minor_locator(NullLocator())
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
a1.set_ylabel("누적 (시작 = 1, 로그)")
a1.text(pd.Timestamp("2016-02-01"), a1.get_ylim()[1] * 0.92, "검증 기간 (기준 고정 뒤 한 번만 계산)", fontsize=9, color="#475569", va="top")
a1.text(pd.Timestamp("2007-04-01"), a1.get_ylim()[1] * 0.92, "학습 기간 (L 고르기)", fontsize=9, color="#475569", va="top")
a1.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
a2.set_ylabel("고점 대비 %")
a1.set_title("미국 ETF 10개 · 매달 판단, 다음 날 시가 체결 · 거래비용 0.1% 포함", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "etf_equity.png")
print("docs/etf_equity.png")
