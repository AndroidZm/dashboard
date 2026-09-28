"""打印 c_exit_slgrid 结果：相对基线的年化差（无压力 / q=0.2 / q=0.3）和回撤。"""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 50)
df = pd.read_csv('c_exit_out/slgrid.csv')
df['cand'] = df.name.str.split('|').str[0]; df['q'] = df.name.str.split('|').str[1]
g = df.groupby(['cand', 'q', 'win'])[['final', 'cagr', 'max_dd']].median().reset_index()
b = g[g.cand == 'base'].set_index(['q', 'win'])
g['dC'] = [c - b.loc[(q, w), 'cagr'] for c, q, w in zip(g.cagr, g.q, g.win)]
g['col'] = g.q + '_' + g.win
p = g.pivot_table(index='cand', columns='col', values='dC').round(2)
dd = g[g.q == 'q0'].pivot_table(index='cand', columns='win', values='max_dd').round(1).add_prefix('dd_')
fin = g[g.q == 'q0'].pivot_table(index='cand', columns='win', values='final').round(0).add_prefix('fin_')
p = pd.concat([p, dd, fin], axis=1)
for q in ['q0', 'q0.2', 'q0.3']:
    p['min_' + q] = p[[f'{q}_A', f'{q}_B', f'{q}_F']].min(axis=1)
print(b.round(2))
print(p.sort_values('min_q0.2', ascending=False).to_string())
