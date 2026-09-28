"""打印 c_exit_final 结果。"""
import pandas as pd, numpy as np
pd.set_option('display.width', 270); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 60)
df = pd.read_csv('c_exit_out/final.csv')
df['drop'] = df['drop'].fillna('')
parts = df.name.str.split('|')
df['cand'] = parts.str[0]
df['tag'] = parts.str[1].fillna('')
df = df.groupby(['cand', 'win', 'drop', 'tag'], as_index=False)[['final', 'cagr', 'max_dd']].median()   # 压力测试按抽样种子取中位
df['k'] = df.win + ':' + df['drop'] + np.where(df.tag != '', ':' + df.tag, '')
b = df[df.cand == 'base'].set_index('k')
df['dC'] = df.cagr - df.k.map(b.cagr)
order = ['base', 'tr_a45_t05_sl40', 't2_a50_t075_x100_t30_sl40', 't2_a50_t075_x100_t35_sl40', 'tr_a50_t10_sl70', 't2_a50_t075_x100_t30_sl70',
         't2_a50_t10_x150_t35_noSL', 'tr_a70_t20_noSL', 'tr_a60_t30_sl70', 'tr_a100_t30_noSL']
short = {c: c.replace('_x100', 'x1').replace('_x150', 'x15')[:18] for c in order}
for v, r in [('final', 0), ('dC', 2), ('max_dd', 1)]:
    print(v)
    print(df.pivot_table(index='k', columns='cand', values=v).reindex(columns=order).rename(columns=short).round(r).to_string())
x = df[(df.cand != 'base')]
s = x.groupby('cand').dC.agg(['min', 'median', 'max'])
s['min_noStress'] = x[~x.tag.str.startswith('q')].groupby('cand').dC.min()
s['min_LOCO_B'] = x[(x.win == 'B') & (x.tag == '')].groupby('cand').dC.min()
print(s.reindex(order[1:]).round(2))
e = pd.read_csv('c_exit_out/final_extra.csv')
print(e.pivot_table(index='name', columns='win', values=['paired_win', 'tr_mean', 'tr_hold', 'pt_mean', 'pt_hold_med', 'pt_win']).reindex(order).round(2).to_string())
print(e.pivot_table(index='name', columns='win', values='dd_month_mode', aggfunc='first').reindex(order).to_string())
