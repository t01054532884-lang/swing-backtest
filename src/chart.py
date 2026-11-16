"""계좌 곡선 그림: docs/equity.png"""

import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS, ROOT
from report import RULES_FIXED_MS
from simulate import portfolio

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
for v, label, color in (("A", "A 진입 알림 기준 (지지·저항 손절)", "#2F5BD3"), ("B", "B 보유 알림 기준 (손절 10%, +10%·+20%)", "#C2410C")):
    t = pd.read_csv(RESULTS / f"trades_{v}.csv")
    _, curve, _ = portfolio(t)
    ax.plot(pd.to_datetime(curve.time, unit="ms"), curve.equity, label=label, color=color, lw=1.6)
ax.axhline(10_000, color="#9AA3B2", lw=0.8)
ax.axvline(pd.to_datetime(RULES_FIXED_MS, unit="ms"), color="#475569", ls="--", lw=0.9)
ax.text(pd.to_datetime(RULES_FIXED_MS, unit="ms"), ax.get_ylim()[1], " 규칙 확정 (2026-09-29)", va="top", fontsize=8, color="#475569")
ax.set_ylabel("가상 계좌 (USDT)")
ax.set_title("관심 29종목 · 거래당 순자산 1% 위험 · 비용(수수료·슬리피지·펀딩) 포함", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
(ROOT / "docs").mkdir(exist_ok=True)
fig.savefig(ROOT / "docs" / "equity.png")
print("docs/equity.png")
