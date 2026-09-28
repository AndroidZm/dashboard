"""v_e_select 检查：同月（及同月同档）随机剔除同样数量信号的安慰剂，80 种子（每个掩码），A/B/B去2024-02；另测“反向规则”（同月剔除最高波动的同样数量）。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
D = data()
t = time.time()
mo = D.sig.signal_date.str[:7].values
tier = D.tier
seeds = range(80)
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
cands = {'C1': (0.38, 1.0, 0.9), 'C2': (0.38, 1.25, 0.9), 'C3': (0.40, 1.5, 1.0)}
cases = [('A', ()), ('B', ()), ('B', ('2024-02',))]
for name, (thr, k, cap) in cands.items():
    rule = VOL60 >= thr
    obs = {c: stats(cfg_of(thr, k, cap), c[0], seeds, c[1])['cagr'] for c in cases}
    # 反向：同月剔除波动最高的同样数量
    anti = np.ones(len(rule), bool)
    for m in np.unique(mo):
        ii = np.nonzero(mo == m)[0]; n = int((~rule[ii]).sum())
        if n:
            anti[ii[np.argsort(-VOL60[ii])[:n]]] = False
    anti_r = {c: stats(replace(cfg_of(None, k, cap), mask=anti), c[0], seeds, c[1])['cagr'] for c in cases}
    for strat in ('month', 'month_tier'):
        rng = np.random.default_rng(2024)
        key = mo if strat == 'month' else np.char.add(mo.astype(str), tier.astype(str))
        res = []
        for i in range(NP):
            mk = np.ones(len(rule), bool)
            for g in np.unique(key):
                ii = np.nonzero(key == g)[0]; n = int((~rule[ii]).sum())
                if n:
                    mk[rng.choice(ii, n, replace=False)] = False
            res.append({c: stats(replace(cfg_of(None, k, cap), mask=mk), c[0], seeds, c[1])['cagr'] for c in cases})
        df = pd.DataFrame(res)
        for c in cases:
            x = df[c].values
            print(f"{name} {strat:10s} {c[0]}{'-'+c[1][0] if c[1] else '':9s} rule {obs[c]:6.2f} anti {anti_r[c]:6.2f} placebo q05/50/95 {np.quantile(x,.05):6.2f}/{np.median(x):6.2f}/{np.quantile(x,.95):6.2f}  P(pl>=rule)={np.mean(x>=obs[c]):.3f}", flush=True)
        a, b = df[cases[0]].values, df[cases[1]].values
        print(f"   joint P(A&B) = {np.mean((a >= obs[cases[0]]) & (b >= obs[cases[1]])):.3f}", flush=True)
print('time', time.time() - t)
