"""v_c_exit 第 6 步：单只股票依赖（逐个剔除对增益贡献最大的信号，基线与候选同时剔除）+ 幸存者压力（独立实现）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_c_exit_lib import *
from engine import run
pd.set_option('display.width', 250)
D = getD()
S = len(D.entry)

def contrib(rule, win, seeds=range(80)):
    pn = np.zeros(S)
    Dx = patched(D, rule)
    for sd in seeds:
        tr = run(Dx, cfg_for(win, record=True), sd)['trades']
        np.add.at(pn, tr.s.values, tr.pnl.values)
    return pn / len(seeds)

rows = []
for cand in ['C1', 'C3', 'C2', 'C4']:
    for win in 'BF':
        d = contrib(CANDS[cand], win) - contrib(CANDS['base'], win)
        top = np.argsort(-d)[:5]
        for k in (1, 3, 5):
            mask = np.ones(S, bool); mask[top[:k]] = False
            m = seeds_stats(D, CANDS[cand], cfg_for(win, mask=mask)); b = seeds_stats(D, CANDS['base'], cfg_for(win, mask=mask))
            rows.append(dict(cand=cand, win=win, drop_top=k, codes=','.join(D.sig.code.values[top[:k]]),
                             top_contrib_wan=round(d[top[:k]].sum() / 1e4, 1), total_diff_wan=round(d.sum() / 1e4, 1),
                             final=m['final'], base=b['final'], dC=m['cagr'] - b['cagr'], dd=m['dd']))
df = pd.DataFrame(rows)
print(df.round(2).to_string(index=False))
df.to_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s6_jack.csv', index=False)

# 幸存者压力（替换模型）：持有期内收盘曾 <= -40% 的信号中抽 q 视为退市；有止损的在 1-sl 出场，无止损按 -95%；出场日不变
# 实现：改写 D.close 的出场日价格（只影响该信号的卖出价与盯市）——为简单起见，直接在副本上把出场日收盘改为 c0*新比率。
import copy
def stressed(rule, q, ss, win):
    t1 = D.day_le(WIN[win][1])
    e, x, r = my_exits(D, rule, t1)
    u = np.random.default_rng(1000 + ss).random(S)
    Dx = patched(D, rule)
    Dx.close = D.close.copy()
    sl = rule.get('sl')
    for s in range(S):
        if x[s] <= e[s] or u[s] >= q:
            continue
        if (D.close[s, e[s] + 1:x[s] + 1] / D.close[s, e[s]]).min() <= 0.6 + 1e-12:
            nr = 0.05 if sl is None else min(r[s], 1 - sl)
            Dx.close[s, x[s]] = D.close[s, e[s]] * nr
    return Dx

rows = []
for q in (0.1, 0.2, 0.3):
    for win in 'BF':
        for cand in ['base', 'C1', 'C3', 'C2']:
            fs = []
            for ss in range(3):
                Dx = stressed(CANDS[cand], q, ss, win)
                fs += [run(Dx, cfg_for(win), sd)['cagr'] for sd in range(40)]
            rows.append(dict(q=q, win=win, cand=cand, cagr=np.median(fs) * 100))
st = pd.DataFrame(rows)
p = st.pivot_table(index=['q', 'win'], columns='cand', values='cagr')
for c in ['C1', 'C3', 'C2']:
    p['d' + c] = p[c] - p['base']
print(p.round(2).to_string())
