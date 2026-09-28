"""e_select s4: 个股层面特征的“同月内置换检验”——把特征标签在同一个月内打乱，看组内收益差是否显著。
同时给出每个特征在 A、B、B 去 2024-02 三个样本的差值，判断方向是否一致。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
rng = np.random.default_rng(0)
flags = {
    'board60': f.board == '60',
    'board00': f.board == '00',
    'board30_68': f.board.isin(['30', '68']),
    'price<=6': f.praw <= 6,
    'price>20': f.praw > 20,
    'price>10': f.praw > 10,
    'bars<500': f.bars < 500,
    'bars<1000': f.bars < 1000,
    'vol60<0.35': f.vol60 < 0.35,
    'vol60>0.6': f.vol60 > 0.6,
    'r250<-0.5': f.r250 < -0.5,
    'r250>-0.2': f.r250 > -0.2,
    'dd250>-0.35': f.dd250 > -0.35,
    'rel250<-0.3': f.rel250 < -0.3,
    'r20<-0.15': f.r20 < -0.15,
    'r60<-0.3': f.r60 < -0.3,
}

def diff_within(g, flag, y):
    # 同月内：flag 组均值 - 非 flag 组均值，只用两组都有样本的月，按月内较小组样本数加权
    num = den = 0.0
    for m, h in g.groupby('month'):
        fl = flag[h.index].values
        if fl.all() or (~fl).all():
            continue
        wgt = min(fl.sum(), (~fl).sum())
        num += wgt * (h[y].values[fl].mean() - h[y].values[~fl].mean())
        den += wgt
    return num / den if den else np.nan

def perm_p(g, flag, y, B=400):
    obs = diff_within(g, flag, y)
    cnt = 0
    months = g.month.values
    fl = flag[g.index].values
    for _ in range(B):
        sh = fl.copy()
        for m in np.unique(months):
            ii = np.nonzero(months == m)[0]
            sh[ii] = rng.permutation(sh[ii])
        if abs(diff_within(g, pd.Series(sh, index=g.index), y)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (B + 1)

samples = {'A': f[f.win == 'A'], 'B': f[f.win == 'B'], 'Bx': f[(f.win == 'B') & (f.month != '2024-02')]}
rows = []
for name, flag in flags.items():
    row = {'feature': name}
    for sn, g in samples.items():
        o, p = perm_p(g, flag, 'ret', B=300)
        row[f'{sn}_n'] = int(flag[g.index].sum())
        row[f'{sn}_dret'] = round(o, 3)
        row[f'{sn}_p'] = round(p, 3)
        row[f'{sn}_dtp'] = round(diff_within(g, flag, 'tp'), 3)
    rows.append(row)
    print(row, flush=True)
out = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(out.to_string(index=False))
out.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/perm.csv', index=False)
