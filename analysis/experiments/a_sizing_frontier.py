"""CAGR-回撤前沿（20 种子中位）：每个回撤上限下各窗口能拿到的最高期末，以及该点的平均仓位。"""
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
O='a_sizing_out/'
rows=[]
for f in ['grid1_A','grid1_B','grid2_A','grid2_B','grid2cap_A','grid2cap_B']:
    d=pd.read_csv(O+f+'.csv'); d['fam']='tier'; d['final']=d.final/1e4
    d['desc']=d.weights.astype(str)+' cap'+d.cap.astype(str)+' pmp'+d.panic_max_pos.fillna(0).astype(int).astype(str)+(' part'+d.partial.astype(str) if 'partial' in d else '')+(' '+d.sizing if 'sizing' in d else '')
    rows.append(d[['win','fam','desc','final','cagr','max_dd','avg_exp']])
d=pd.read_csv(O+'slots.csv'); d['fam']='slots'; d['final']=d.final/1e4
d['desc']='N'+d.slots.astype(str)+' cap'+d.cap.astype(str)+' part'+d.partial.astype(str)+' '+d.sizing
rows.append(d[['win','fam','desc','final','cagr','max_dd','avg_exp']])
d=pd.read_csv(O+'adaptive.csv'); d['desc']=d.fam+' '+d.p.astype(str)
rows.append(d[['win','fam','desc','final','cagr','max_dd','avg_exp']])
X=pd.concat(rows,ignore_index=True)
for w in 'ABF':
    x=X[X.win==w]
    print('=== window',w,'n',len(x))
    for lim in [-0.20,-0.225,-0.25,-0.275,-0.30,-0.325,-0.35,-0.40,-0.45,-1]:
        y=x[x.max_dd>=lim]
        if len(y)==0: continue
        r=y.loc[y.final.idxmax()]
        print(f' dd>={lim:6.3f}: {r.final:6.1f}万 cagr {r.cagr*100:5.2f}% dd {r.max_dd*100:5.1f}% exp {r.avg_exp*100:4.0f}%  [{r.fam}] {r.desc}')
