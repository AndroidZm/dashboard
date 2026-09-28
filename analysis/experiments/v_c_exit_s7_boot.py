"""v_c_exit 第 7 步：样本构成稳健性——随机剔除 10% / 20% 信号（基线与候选用同一 mask），40 次 x 20 种子，看年化差分布；
另：幸存者压力多抽样（10 次抽样 x 20 种子）。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_c_exit_lib import *
from engine import run
from multiprocessing import Pool
D = getD()
S = len(D.entry)
NAMES = ['base', 'C1', 'C2', 'C3', 'C4']

def job(a):
    frac, k, win = a
    mask = np.random.default_rng(500 + k).random(S) >= frac
    out = dict(frac=frac, k=k, win=win)
    for n in NAMES:
        m = seeds_stats(D, CANDS[n], cfg_for(win, mask=mask), range(20))
        out[n] = m['cagr']; out['dd_' + n] = m['dd']; out['fin_' + n] = m['final']
    return out

def stress_job(a):
    q, ss, win = a
    t1 = D.day_le(WIN[win][1])
    u = np.random.default_rng(2000 + ss).random(S)
    out = dict(q=q, ss=ss, win=win)
    for n in NAMES:
        rule = CANDS[n]
        e, x, r = my_exits(D, rule, t1)
        Dx = patched(D, rule); Dx.close = D.close.copy()
        sl = rule.get('sl')
        for s in range(S):
            if x[s] <= e[s] or u[s] >= q:
                continue
            if (D.close[s, e[s] + 1:x[s] + 1] / D.close[s, e[s]]).min() <= 0.6 + 1e-12:
                Dx.close[s, x[s]] = D.close[s, e[s]] * (0.05 if sl is None else min(r[s], 1 - sl))
        out[n] = np.median([run(Dx, cfg_for(win), sd)['cagr'] for sd in range(20)]) * 100
    return out

if __name__ == '__main__':
    t = time.time()
    for n in NAMES:
        for w in 'ABF':
            my_exits(D, CANDS[n], D.day_le(WIN[w][1]))
    jobs = [(f, k, w) for f in (0.1, 0.2) for k in range(40) for w in 'ABF']
    with Pool(3) as p:
        df = pd.DataFrame(p.map(job, jobs))
    df.to_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s7_boot.csv', index=False)
    for c in NAMES[1:]:
        df['d' + c] = df[c] - df['base']
    pd.set_option('display.width', 250)
    g = df.groupby(['frac', 'win'])
    for c in NAMES[1:]:
        print(c, '\n', g['d' + c].describe(percentiles=[.1, .5, .9]).round(2).assign(pos=g['d' + c].apply(lambda s: (s > 0).mean()).round(2)).to_string())
    jobs = [(q, ss, w) for q in (0.2, 0.3) for ss in range(10) for w in 'BF']
    with Pool(3) as p:
        st = pd.DataFrame(p.map(stress_job, jobs))
    st.to_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s7_stress.csv', index=False)
    for c in NAMES[1:]:
        st['d' + c] = st[c] - st['base']
    print(st.groupby(['q', 'win'])[['dC1', 'dC2', 'dC3', 'dC4']].describe(percentiles=[.1, .5, .9]).round(2).T.to_string())
    print('elapsed', time.time() - t)
