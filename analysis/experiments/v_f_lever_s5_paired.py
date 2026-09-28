"""v_f_lever s5: 逐种子配对比较（同种子 = 同成交顺序）+ 融资发生时点（是否落在 2010-03-31 两融开通前）。"""
import os, sys
import numpy as np, pandas as pd
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v_f_lever_lib import Data, Lcfg, per_seed, run, mmask

D = Data.load()
for w in "BAF":
    b = per_seed(D, Lcfg(w, 1.0))
    for L in [1.1, 1.5, 2.2]:
        for tag, kw in [("", {}), ("剔2024-02", dict(mask=mmask(D, ["2024-02"]))), ("次日+滑点", dict(lag=1, slip=0.005))]:
            bb = b if not kw else per_seed(D, Lcfg(w, 1.0, **kw))
            x = per_seed(D, Lcfg(w, L, borrow_rate=0.08, **kw))
            ratio = x.final / bb.final
            ddd = (x.max_dd - bb.max_dd) * 100
            print(f"{w} L{L} {tag:8s}: 配对期末比 中位 {ratio.median():.3f} min {ratio.min():.3f} max {ratio.max():.3f} 胜出种子比例 {(ratio>1).mean():.2f}"
                  f" | 回撤变化 pp 中位 {ddd.median():.1f} 最差 {ddd.min():.1f}")
# 融资发生时点
for L in [1.5, 2.2]:
    r = run(D, replace(Lcfg("F", L, borrow_rate=0.08), record=True), 13)
    eq = r["equity"]; t0 = D.day("2006-01-04")
    pos = np.zeros(len(eq))
    for s, bt, st, amt in r["trades"][["s", "buy_t", "sell_t", "amount"]].itertuples(index=False):
        pos[bt - t0: st - t0] += amt / D.close[s, bt] * D.close[s, bt:st]
    debt = pd.Series(np.maximum(pos - eq.values, 0), index=eq.index)
    pre = debt[debt.index < "2010-03-31"]; mid = debt[(debt.index >= "2010-03-31") & (debt.index < "2015-11-23")]
    print(f"F seed13 L{L}: 两融开通前(2010-03-31前)有负债天数 {(pre>1).sum()} 最大负债/权益 {(pre/eq[pre.index]).max():.2f};"
          f" 2010-03~2015-11 有负债天数 {(mid>1).sum()}; 全期有负债天数 {(debt>1).sum()}")
