"""跨窗口选择：在一个窗口按风险档选最优（20 种子），拿到另一个窗口看；两个方向都做。"""
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
O='a_sizing_out/'
def load(w):
    parts=[]
    for f,extra in [('grid1',{}),('grid2',{}),('grid2cap',{}),('slots',{})]:
        d=pd.read_csv(O+(f'{f}_{w}.csv' if f!='slots' else 'slots.csv'))
        if f=='slots': d=d[d.win==w]
        parts.append(d)
    d=pd.concat(parts,ignore_index=True)
    for c,dv in [('partial',False),('sizing','cost'),('slots',np.nan)]:
        if c not in d: d[c]=dv
        d[c]=d[c].fillna(dv) if c!='slots' else d[c]
    d['pmp']=d.panic_max_pos.fillna(0).astype(int)
    d['weights']=d.weights.fillna('-')
    d['key']=d.weights.astype(str)+'|'+d.slots.fillna(0).astype(int).astype(str)+'|'+d.cap.astype(str)+'|'+d.pmp.astype(str)+'|'+d.partial.astype(str)+'|'+d.sizing
    d['final']=d.final/1e4
    return d.drop_duplicates('key')
A,B=load('A'),load('B')
M=A.merge(B,on='key',suffixes=('_A','_B'))
print('n configs',len(M), 'spearman final', round(M.final_A.corr(M.final_B,method='spearman'),3))
base={'A':(130.7,-0.186),'B':(173.5,-0.268)}
print('baseline A 130.7/-18.6  B 173.5/-26.8 (80 seeds)')
for src,dst in [('A','B'),('B','A')]:
    for lim in [-0.30,-0.40,-1.0]:
        d=M[M['max_dd_'+src]>=lim].sort_values('final_'+src,ascending=False)
        top=d.head(1).iloc[0]; top10=d.head(10)
        print(f'select on {src} dd>={lim}: {top.key}  {src}:{top["final_"+src]:.1f}/{top["max_dd_"+src]:.3f} -> {dst}:{top["final_"+dst]:.1f}/{top["max_dd_"+dst]:.3f}  '
              f'| top10 median in {dst}: {top10["final_"+dst].median():.1f} (range {top10["final_"+dst].min():.1f}-{top10["final_"+dst].max():.1f}), dd {top10["max_dd_"+dst].median():.3f}; '
              f'rank pct in {dst}: {(M["final_"+dst]<top["final_"+dst]).mean():.2f}')
