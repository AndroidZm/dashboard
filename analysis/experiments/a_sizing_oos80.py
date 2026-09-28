"""80 种子结果上的跨窗口选择（plateau 网格 80 组 + 消融 20 组 + 确认候选），按三档风险在一个窗口选、另一个窗口看。"""
import pandas as pd, numpy as np
O='a_sizing_out/'
p=pd.read_csv(O+'plateau.csv'); p['name']='w'+p.weights.astype(str)+' cap1 part'; p['final']=p.final/1e4
a=pd.read_csv(O+'ablation.csv'); a['final']=a.final/1e4
c=pd.read_csv(O+'confirm.csv'); c=c[c['drop'].isna()]
X=pd.concat([p[['name','win','final','cagr','max_dd']],a[['name','win','final','cagr','max_dd']],c[['name','win','final','cagr','max_dd']]],ignore_index=True).drop_duplicates(['name','win'])
W=X.pivot_table(index='name',columns='win',values=['final','max_dd','cagr'])
W=W.dropna()
print('n configs',len(W), 'spearman A-B final:',round(W[('final','A')].corr(W[('final','B')],method='spearman'),3))
base={'A':130.66,'B':173.54}
for src,dst in [('A','B'),('B','A')]:
    for lim in [-0.30,-0.40,-1.0]:
        d=W[W[('max_dd',src)]>=lim].sort_values(('final',src),ascending=False)
        t=d.iloc[0]; t5=d.head(5)
        print(f'[{src}->{dst}] dd>={lim}: {d.index[0]} | {src} {t[("final",src)]:.1f}/{t[("max_dd",src)]*100:.1f}% -> {dst} {t[("final",dst)]:.1f}/{t[("max_dd",dst)]*100:.1f}% (base {base[dst]}) '
              f'| top5 in {dst}: med {t5[("final",dst)].median():.1f} min {t5[("final",dst)].min():.1f} max {t5[("final",dst)].max():.1f}')
