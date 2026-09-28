"""e_select s6: 用信号日可得特征剔除信号（mask），组合层面 A / B / B去2024-02 对比基线（20 种子中位）。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
import numpy as np, pandas as pd
from e_select_lib import *
D = data()
f = pd.read_pickle('/home/user/dashboard/analysis/experiments/e_select_out/feat.pkl')
T = D.tier
masks = {
    'baseline': np.ones(len(f), bool),
    'no_board60': f.board.values != '60',
    'no_price>20': f.praw.values <= 20,
    'no_price>10': f.praw.values <= 10,
    'no_price<3': f.praw.values >= 3,
    'no_bars<500': f.bars.values >= 500,
    'no_bars<1000': f.bars.values >= 1000,
    'no_vol60>0.6': f.vol60.values <= 0.6,
    'no_vol60<0.35': f.vol60.values >= 0.35,
    'no_rel250<-0.3': f.rel250.values >= -0.3,
    'no_r250<-0.5': f.r250.values >= -0.5,
    'no_dd250>-0.35': f.dd250.values <= -0.35,
    'skip_idxdd>-0.10': f.idx_dd250.values <= -0.10,
    'skip_idxdd>-0.15': f.idx_dd250.values <= -0.15,
    'skip_idxdd>-0.20': f.idx_dd250.values <= -0.20,
    'skip_idxdd>-0.25': f.idx_dd250.values <= -0.25,
    'skip_t0_idxdd>-0.15': (T != 0) | (f.idx_dd250.values <= -0.15),
    'skip_t0_idxdd>-0.20': (T != 0) | (f.idx_dd250.values <= -0.20),
    'skip_hsdd>-0.10': f.hs_dd250.values <= -0.10,
    'skip_tier0': T != 0,
}
rows = []
base = {}
for name, mk in masks.items():
    r = {'rule': name, 'n_sig_A': int(mk[f.win.values == 'A'].sum()), 'n_sig_B': int(mk[f.win.values == 'B'].sum())}
    for win, drop in [('A', None), ('B', None), ('B', '2024-02')]:
        m = ev(Config(mask=mk), win, range(20), drop)
        key = win + ('x' if drop else '')
        r[key + '_cagr'] = m['cagr'] * 100
        r[key + '_dd'] = m['max_dd'] * 100
        r[key + '_final'] = m['final'] / 1e4
    rows.append(r)
df = pd.DataFrame(rows)
for k in ['A', 'B', 'Bx']:
    df[k + '_dC'] = df[k + '_cagr'] - df.loc[0, k + '_cagr']
    df[k + '_dDD'] = df[k + '_dd'] - df.loc[0, k + '_dd']
pd.set_option('display.width', 250)
print(df[['rule', 'n_sig_A', 'n_sig_B', 'A_cagr', 'A_dC', 'A_dDD', 'B_cagr', 'B_dC', 'B_dDD', 'Bx_cagr', 'Bx_dC', 'Bx_dDD']].round(2).to_string(index=False))
df.to_csv('/home/user/dashboard/analysis/experiments/e_select_out/s6_masks.csv', index=False)
