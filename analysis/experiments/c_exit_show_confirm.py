"""打印 c_exit_confirm / c_exit_confirm2 的结果（相对同口径基线的年化差）。"""
import sys
import pandas as pd, numpy as np
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 50)
f = sys.argv[1] if len(sys.argv) > 1 else 'c_exit_out/confirm.csv'
df = pd.read_csv(f)
df['drop'] = df['drop'].fillna('')
df['stress'] = df.name.str.split('|').str[1].fillna('')
df['cand'] = df.name.str.split('|').str[0]
df['k'] = df.win + ':' + df['drop'] + np.where(df.stress != '', ':' + df.stress, '')
df = df.drop_duplicates(['cand', 'k'])
b = df[df.cand == df.cand.iloc[0]].set_index('k')
df['dC'] = df.cagr - df.k.map(b.cagr); df['dDD'] = df.max_dd - df.k.map(b.max_dd)
order = list(dict.fromkeys(df.cand))
short = {c: c.replace('base_', '').replace('_noSL', '_nS')[:16] for c in order}
for v, r in [('final', 0), ('dC', 2), ('max_dd', 1)]:
    print(v)
    print(df.pivot_table(index='k', columns='cand', values=v).reindex(columns=order).rename(columns=short).round(r).to_string())
print(df[df.cand != order[0]].groupby('cand').dC.agg(['min', 'median', 'max']).reindex(order[1:]).round(2))
