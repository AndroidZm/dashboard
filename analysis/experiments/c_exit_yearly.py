"""逐年分解：候选规则与基线在同一执行种子下的年度收益差（80 种子中位），看增益来自哪几年。"""
import sys
from c_exit_lib import *
from c_exit_confirm import CANDS
if __name__ == "__main__":
    D = getD()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["base_tp40_sl40", "fx_tp40_noSL", "tr_a50_t10_sl40", "tr_a50_t10_noSL", "tr_a70_t20_noSL"]
    win = sys.argv[2] if len(sys.argv) > 2 else "F"
    out = {}
    finals = {}
    for n in names:
        ys = []
        fs = []
        for seed in range(80):
            r = run_legs(D, replace(cfg_for(win), record=True), CANDS[n], seed)
            eq = r["equity"]
            y = eq.groupby(eq.index.str[:4]).last()
            y0 = pd.concat([pd.Series([600000.0], index=["start"]), y])
            ys.append(y0.pct_change().iloc[1:])
            fs.append(r["final"])
        out[n] = pd.concat(ys, axis=1).median(axis=1) * 100
        finals[n] = np.array(fs)
    df = pd.DataFrame(out)
    b = names[0]
    for n in names[1:]:
        df["d_" + n] = df[n] - df[b]
    pd.set_option("display.width", 250)
    print(df.round(1).to_string())
    for n in names:
        f = finals[n] / 1e4
        print(n, "final quantiles p10/p25/p50/p75/p90:", np.percentile(f, [10, 25, 50, 75, 90]).round(0), " win-rate vs base same seed:", np.mean(finals[n] > finals[b]).round(2))
