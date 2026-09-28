"""v_c_exit 第 8 步：对称截尾（同时剔除候选相对基线贡献最大的 k 笔和最差的 k 笔）+ A 上选出的参数在 B 的 80 种子结果。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_c_exit_lib import *
from engine import run
pd.set_option('display.width', 250)
D = getD(); S = len(D.entry)

def contrib(rule, win):
    pn = np.zeros(S); Dx = patched(D, rule)
    for sd in range(80):
        tr = run(Dx, cfg_for(win, record=True), sd)['trades']
        np.add.at(pn, tr.s.values, tr.pnl.values)
    return pn / 80
rows = []
for cand in ['C1', 'C3', 'C2']:
    for win in 'BF':
        d = contrib(CANDS[cand], win) - contrib(CANDS['base'], win)
        o = np.argsort(-d)
        for k in (1, 3, 5, 10):
            drop = np.r_[o[:k], o[-k:]]
            mask = np.ones(S, bool); mask[drop] = False
            m = seeds_stats(D, CANDS[cand], cfg_for(win, mask=mask)); b = seeds_stats(D, CANDS['base'], cfg_for(win, mask=mask))
            rows.append(dict(cand=cand, win=win, k=k, gain_top=round(d[o[:k]].sum() / 1e4, 1), loss_bot=round(d[o[-k:]].sum() / 1e4, 1),
                             final=m['final'], base=b['final'], dC=m['cagr'] - b['cagr']))
print(pd.DataFrame(rows).round(2).to_string(index=False))
# A 上选出的配置 -> B（80 种子）
T = lambda a, w, sl: dict(kind='trail', act=a, trail=w, sl=sl, tp=None)
T2 = lambda a, w, a2, w2, sl: dict(kind='trail2', act=a, trail=w, act2=a2, trail2=w2, sl=sl, tp=None)
picks = {'A-pick trail(all sl)': T(0.45, 0.3, 0.7), 'A-pick trail sl0.4': T(0.5, 0.3, 0.4), 'A-pick trail2 (act free)': T2(0.6, 0.1, 0.8, 0.3, 0.7),
         'A-pick trail2 act=0.5 fixed': T2(0.5, 0.1, 0.8, 0.3, 0.7), 'A-pick fixed': dict(kind='fixed', tp=1.0, sl=0.4)}
for n, r in picks.items():
    out = {w: seeds_stats(D, r, cfg_for(w)) for w in 'ABF'}
    bb = {w: seeds_stats(D, CANDS['base'], cfg_for(w)) for w in 'ABF'}
    print(n, ' '.join(f"{w}: {out[w]['final']:.1f}万 {out[w]['cagr']:.2f}% dC {out[w]['cagr']-bb[w]['cagr']:+.2f} dd {out[w]['dd']:.1f}" for w in 'ABF'))
