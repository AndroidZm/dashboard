"""e_select 公共特征：每个信号在信号日可得的特征 + 逐笔结果（±40%，全样本到 2026-08-21）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis')
import numpy as np
import pandas as pd
from engine import Data, Config, run_seeds, run

BIG = ['2012-01', '2018-02', '2022-05', '2022-10', '2024-02']
WA = ('2006-01-04', '2016-01-28')
WB = ('2016-01-01', '2026-08-21')


def features(D, lag=0):
    sig = D.sig
    S = len(sig)
    e_arr, x_arr, ratio = D.exits(0.4, 0.4, None, None, lag)
    f = pd.DataFrame(index=range(S))
    f['date'] = sig.signal_date.values
    f['month'] = f.date.str[:7]
    f['tier'] = D.tier
    f['n30'] = D.n30
    f['bars'] = sig.bars_before_signal.values
    f['praw'] = sig.buy_price_raw.values
    f['board'] = sig.code.str[:2].values
    e = D.entry.astype(int)
    C = D.close
    for k in (20, 60, 120, 250):
        f[f'r{k}'] = C[np.arange(S), e] / C[np.arange(S), np.maximum(e - k, 0)] - 1
    f['dd250'] = [C[s, e[s]] / C[s, max(e[s] - 250, 0):e[s] + 1].max() - 1 for s in range(S)]
    ix = D.index['sh000852'].values
    f['idx_dd250'] = [ix[t] / ix[max(t - 250, 0):t + 1].max() - 1 for t in e]
    f['idx_r20'] = [ix[t] / ix[max(t - 20, 0)] - 1 for t in e]
    f['idx_r250'] = [ix[t] / ix[max(t - 250, 0)] - 1 for t in e]
    hs = D.index['sh000300'].values
    f['hs_dd250'] = [hs[t] / hs[max(t - 250, 0):t + 1].max() - 1 for t in e]
    # 股票相对指数的 60/250 日超额
    f['rel60'] = f.r60 - np.array([ix[t] / ix[max(t - 60, 0)] - 1 for t in e])
    f['rel250'] = f.r250 - f.idx_r250
    # 波动率（60 日日收益标准差）
    lr = np.diff(np.log(C), axis=1)
    f['vol60'] = [lr[s, max(e[s] - 60, 0):e[s]].std() * np.sqrt(244) for s in range(S)]
    # 结果
    f['ratio'] = ratio
    f['ret'] = ratio - 1
    f['held'] = (x_arr - e_arr)
    f['tp'] = (ratio >= 1.4).astype(int)
    f['sl'] = (ratio <= 0.6).astype(int)
    f['open'] = 1 - f.tp - f.sl
    f['yrs'] = f.held / 244.0
    f['win'] = np.where(f.date < '2016-01-01', 'A', 'B')
    return f


def bucket_stats(f, col, bins=None, labels=None):
    g = f.copy()
    if bins is not None:
        g['b'] = pd.cut(g[col], bins=bins, labels=labels)
    else:
        g['b'] = g[col]
    # 同月去均值（组内比较）
    g['ret_dm'] = g.ret - g.groupby('month').ret.transform('mean')
    g['tp_dm'] = g.tp - g.groupby('month').tp.transform('mean')
    rows = []
    for (w, b), h in g.groupby(['win', 'b'], observed=True):
        rows.append(dict(win=w, b=b, n=len(h), nclus=h.month.nunique(), tp=h.tp.mean(), sl=h.sl.mean(),
                         ret=h.ret.mean(), held=h.held.median(), eff=h.ret.sum() / h.yrs.sum(),
                         ret_dm=h.ret_dm.mean(), tp_dm=h.tp_dm.mean()))
    return pd.DataFrame(rows)
