"""e_select 最终规则（可独立复用）：剔除“信号日前 60 个交易日年化波动率 < 阈值”的慢信号，可选配合权重放大 k / 总仓位上限。

    from e_select_rule import vol60, cands
    D = Data.load(); v = vol60(D); cfg = cands(v)['C2']
"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis')
import numpy as np
from engine import Data, Config, run_seeds


def vol60(D, k=60):
    """信号日（含当日）之前 k 个交易日的日对数收益标准差 x sqrt(244)；只用有成交的日子（停牌前向填充的零收益剔除）。"""
    e = D.entry.astype(int)
    lr = np.diff(np.log(D.close), axis=1)
    out = np.empty(len(e))
    for s in range(len(e)):
        a = max(e[s] - k, 0)
        seg = lr[s, a:e[s]]
        tr = D.traded[s, a + 1:e[s] + 1]
        seg = seg[tr] if tr.sum() > 5 else seg
        out[s] = seg.std() * np.sqrt(244)
    return out


def cands(v):
    return {
        'C1_vol38': Config(mask=v >= 0.38),
        'C2_vol38_k1.25': Config(mask=v >= 0.38, weights=(0.025, 0.10, 0.075)),
        'C3_vol40_k1.5_cap1.0': Config(mask=v >= 0.40, weights=(0.03, 0.12, 0.09), cap=1.0),
    }


if __name__ == '__main__':
    from dataclasses import replace
    D = Data.load()
    v = vol60(D)
    V = np.load('/home/user/dashboard/analysis/experiments/e_select_out/vols.npy')
    print('matches s7 vol60:', np.allclose(v, V[1]))
    for n, c in {'baseline': Config(), **cands(v)}.items():
        for a, b in [('2016-01-01', '2026-08-21'), ('2006-01-04', '2016-01-28'), ('2006-01-04', '2026-08-21')]:
            m = run_seeds(D, replace(c, start=a, end=b))
            print(f"{n:22s} {a}~{b}: final {m['final']/1e4:6.1f}万 CAGR {m['cagr']*100:5.2f}% maxDD {m['max_dd']*100:5.1f}% trades {m['n_trades']:.0f} exp {m['avg_exposure']*100:.0f}%")
