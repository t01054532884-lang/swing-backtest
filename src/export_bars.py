"""웹 화면에서 거래를 차트로 보여 주려고 1시간봉만 한 파일로 묶는다: results/bars_1h.csv.gz"""

import pandas as pd

from common import RESULTS, config, load

parts = []
for symbol in config()["symbols"]:
    df = load(symbol, "1h")[["open_time", "open", "high", "low", "close"]]
    df.insert(0, "symbol", symbol)
    parts.append(df)
out = pd.concat(parts)
out.to_csv(RESULTS / "bars_1h.csv.gz", index=False, float_format="%.6g")
print(f"results/bars_1h.csv.gz {len(out):,}행")
