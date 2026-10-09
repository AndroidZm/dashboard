"""i_nodip_: “同日信号按跌幅深浅排优先级”这个发现的稳健性。
同族特征（ma60/ret60/ret20/dd250/vol20/amp20）是否都有效（平台还是尖峰）；分时段；逐簇剔除；实盘口径；叠加在稳健方案（单笔×1.1、上限 99%）上。
输出 i_nodip_out/priority_robust.txt / .csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Config, Data, run_seeds  # noqa: E402

OUT = os.path.join(HERE, "i_nodip_out")


def main():
    D = Data.load()
    F = pd.read_csv(os.path.join(OUT, "screen_score.csv"), dtype={"code": str, "code3": str})
    f = lambda c: F[c].values.astype(float)  # noqa: E731
    pri = {
        "随机(基线)": None,
        "ma120 低者先": -f("ma120"),
        "ma120 高者先(反向)": f("ma120"),
        "ma60 低者先": -f("ma60"),
        "ret60 低者先": -f("ret60"),
        "ret20 低者先": -f("ret20"),
        "dd250 低者先": -f("dd250"),
        "dd_all 低者先": -f("dd_all"),
        "vol20 高者先": f("vol20"),
        "amp20 高者先": f("amp20"),
        "ret1 高者先(信号日涨幅)": f("ret1"),
        "价格低者先": -f("log_price"),
        "上市久者先": f("bars_before_signal"),
        "个股分高者先": f("个股分"),
        "样本外模型分高者先": f("nodip0_个股_logit"),
    }
    ym = F.ym.values
    cases = {
        "B": dict(start="2016-01-01"), "A": dict(start="2006-01-04", end="2016-01-28"), "F": dict(start="2006-01-04"),
        "B1 2016-2020": dict(start="2016-01-01", end="2020-12-31"), "B2 2021-2026": dict(start="2021-01-01"),
    }
    loco = ["2012-01", "2018-02", "2022-05", "2022-10", "2024-02"]
    rows = []
    for pn, p in pri.items():
        for cn, cd in cases.items():
            for ex_name, ex in {"当日": {}, "实盘": dict(lag=1, slip=0.005)}.items():
                r = run_seeds(D, Config(**cd, priority=p, **ex))
                rows.append(dict(优先级=pn, case=cn, exec=ex_name, cagr=r["cagr"] * 100, dd=r["max_dd"] * 100, n=r["n_trades"], p10=r["cagr_p10"] * 100, p90=r["cagr_p90"] * 100))
        for m in loco:
            r = run_seeds(D, Config(start="2006-01-04", priority=p, mask=ym != m))
            rows.append(dict(优先级=pn, case=f"F去{m}", exec="当日", cagr=r["cagr"] * 100, dd=r["max_dd"] * 100, n=r["n_trades"], p10=r["cagr_p10"] * 100, p90=r["cagr_p90"] * 100))
        # 叠加在稳健方案上
        for cn, cd in {"B": dict(start="2016-01-01"), "F": dict(start="2006-01-04")}.items():
            for ex_name, ex in {"当日": {}, "实盘": dict(lag=1, slip=0.005)}.items():
                r = run_seeds(D, Config(**cd, priority=p, weights=(0.022, 0.088, 0.066), cap=0.99, **ex))
                rows.append(dict(优先级=pn, case=f"{cn}+稳健方案(×1.1,99%)", exec=ex_name, cagr=r["cagr"] * 100, dd=r["max_dd"] * 100, n=r["n_trades"], p10=r["cagr_p10"] * 100, p90=r["cagr_p90"] * 100))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "priority_robust.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    with open(os.path.join(OUT, "priority_robust.txt"), "w") as fh:
        for e in ("当日", "实盘"):
            d = R[R.exec == e].assign(v=lambda d: d.cagr.map("{:.2f}".format) + "/" + d.dd.map("{:.1f}".format))
            t = d.pivot(index="优先级", columns="case", values="v").reindex(list(pri))
            order = [c for c in ["A", "B", "F", "B1 2016-2020", "B2 2021-2026"] + [f"F去{m}" for m in loco] + ["B+稳健方案(×1.1,99%)", "F+稳健方案(×1.1,99%)"] if c in t.columns]
            s = f"\n===== {e}口径：年化%/最大回撤%（80 种子中位）\n" + t[order].to_string()
            print(s)
            fh.write(s + "\n")
        d = R[(R.exec == "当日") & (R.case.isin(["A", "B", "F"]))]
        s = "\n===== 种子离散（当日）：年化 p10 / 中位 / p90\n" + d.assign(v=lambda d: d.p10.map("{:.2f}".format) + "/" + d.cagr.map("{:.2f}".format) + "/" + d.p90.map("{:.2f}".format)).pivot(index="优先级", columns="case", values="v").reindex(list(pri)).to_string()
        print(s)
        fh.write(s + "\n")


if __name__ == "__main__":
    main()
