"""s8: 如果照“逐笔 Kelly”下注（每笔 183% / 半 Kelly 92% / 1/4 Kelly 46% 权益），即便总仓位封顶 2 倍、融资 8%、
130% 强平，会怎样。80 种子。输出 f_lever_out/s8_pertrade_kelly.txt"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f_lever_engine import Data, cfg_win, run_seeds  # noqa: E402

D = Data.load()
for w in "BAF":
    for f in [1.83, 0.915, 0.46]:
        m = run_seeds(D, cfg_win(w, weights=(f, f, f), cap=2.0, borrow_rate=0.08, margin_call=(1.3, 1.5),
                                 sizing="mtm", cap_basis="mtm"), range(80))
        print(w, "per-trade f", f, round(m["final"] / 1e4, 1), round(m["cagr"] * 100, 2), "dd", round(m["max_dd"] * 100, 1),
              "p10", round(m["dd_p10"] * 100, 1), "worst", round(m["dd_worst"] * 100, 1), "mc_any", m["mc_any"],
              "n", m["n_trades"], "min final", round(m["final_min"] / 1e4, 1))
