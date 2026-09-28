"""v_e_select 检查：逐笔机制（波动 vs 持有期/胜率/年化资金收益）、0.40~0.42 断崖处的信号、按档位拆分剔除。"""
import sys
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_e_select_lib import *
from scipy.stats import spearmanr
D = data()
e, x, ratio = D.exits(0.4, 0.4)
sig = D.sig
f = pd.DataFrame(dict(date=sig.signal_date.values, mo=sig.signal_date.str[:7].values, tier=D.tier, v=VOL60, held=x - e, ret=ratio - 1,
                      tp=(ratio >= 1.4).astype(int), sl=(ratio <= 0.6).astype(int)))
f['win'] = np.where(f.date < '2016-01-01', 'A', 'B')
f['lo'] = f.v < 0.38
f['ret_dm'] = f.ret - f.groupby('mo').ret.transform('mean')
f['held_dm'] = f.held - f.groupby('mo').held.transform('mean')
for w, g in f.groupby('win'):
    print(w, 'spearman(v,held)=%.3f' % spearmanr(g.v, g.held)[0], 'within-month spearman(v,held_dm)=%.3f' % spearmanr(g.v, g.held_dm)[0])
    print(g.groupby('lo').agg(n=('v', 'size'), tp=('tp', 'mean'), sl=('sl', 'mean'), ret=('ret', 'mean'), ret_dm=('ret_dm', 'mean'), held_med=('held', 'median'),
                             ret_per_yr=('ret', lambda r: r.sum()), yrs=('held', lambda h: h.sum() / 244)).assign(eff=lambda d: d.ret_per_yr / d.yrs).round(3).to_string())
    gx = g[g.mo != '2024-02']
    print('  excl 2024-02:', gx.groupby('lo').agg(n=('v', 'size'), tp=('tp', 'mean'), held_med=('held', 'median'), ret_dm=('ret_dm', 'mean')).round(3).to_string().replace('\n', ' | '))
cl = f[(f.v >= 0.40) & (f.v < 0.42)]
print('signals with 0.40<=v<0.42:', len(cl)); print(cl.groupby(['mo', 'tier']).size().to_string())
# 只剔除非恐慌档的低波动 / 只剔除恐慌档的低波动
for name, mk in {'lo_only_t01': ~((VOL60 < 0.38) & (D.tier < 2)), 'lo_only_t2': ~((VOL60 < 0.38) & (D.tier == 2))}.items():
    for w, drop in (('A', ()), ('B', ()), ('B', ('2024-02',))):
        m = stats(replace(cfg_of(), mask=mk), w, drop=drop); m0 = stats(cfg_of(), w, drop=drop)
        print(name, w, drop, 'd=%.2f' % (m['cagr'] - m0['cagr']))
print('--- within-month held_dm / ret_dm by lo, per window and tier group')
f['tg'] = np.where(f.tier == 2, 'panic', 'norm/adj')
print(f.groupby(['win', 'tg', 'lo']).agg(n=('v', 'size'), held_dm=('held_dm', 'mean'), ret_dm=('ret_dm', 'mean'), held_med=('held', 'median'), tp=('tp', 'mean')).round(3).to_string())
print('months contributing removed signals (B):'); print(f[(f.win == 'B') & f.lo].groupby('mo').size().sort_values(ascending=False).head(12).to_string())
