"""密度自适应的仓位规则（窗口 A、B 冲突的可能解）：
  cashfrac f : 单笔 = f × 当时现金（信号稀疏时单笔大、扎堆时几何递减）
  n30 N0,b   : 单笔权重 = 1/(N0 + b·n30)，n30 = 信号日前 30 天全表信号数（档位的连续版）
cap=1.0、partial=True、不限恐慌持仓数。"""
from a_sizing_lib import *
from functools import partial as fpartial
import time, sys

def wf_cashfrac(s, t, st, f):
    return f * max(st["cash"], 0.0) / st["equity"]

def wf_n30(s, t, st, N0, b):
    return 1.0 / (N0 + b * D().n30[s])

def params():
    P=[]
    for f in [0.03,0.05,0.07,0.1,0.12,0.15,0.18,0.2,0.25,0.3,0.4]:
        P.append(dict(weight_fn=fpartial(wf_cashfrac,f=f),cap=1.0,partial=True,panic_max_pos=None,_fam='cashfrac',_p=f))
    for N0 in [3,4,5,6,8,10,12,15,20,25]:
        for b in [0.05,0.1,0.2,0.3,0.5,1.0]:
            P.append(dict(weight_fn=fpartial(wf_n30,N0=N0,b=b),cap=1.0,partial=True,panic_max_pos=None,_fam='n30',_p=f'{N0},{b}'))
    return P

def _ev(args):
    p,w,drop,ns=args
    q={k:v for k,v in p.items() if not k.startswith('_')}
    m=run_seeds(D(),cfg_for(q,w,drop),seeds=range(ns))
    return dict(fam=p['_fam'],p=p['_p'],win=w,drop='|'.join(drop),final=m['final']/1e4,cagr=m['cagr'],max_dd=m['max_dd'],dd_p10=m['dd_p10'],
                avg_exp=m['avg_exposure'],n=m['n_trades'],n0=m['n0'],n1=m['n1'],n2=m['n2'])

if __name__=='__main__':
    from multiprocessing import Pool
    D()
    jobs=[(p,w,(),20) for p in params() for w in ['A','B','F']]
    t=time.time()
    with Pool(3) as pool: rows=pool.map(_ev,jobs,chunksize=4)
    df=pd.DataFrame(rows); df.to_csv('a_sizing_out/adaptive.csv',index=False); print(time.time()-t)
    pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
    print(df.pivot_table(index=['fam','p'],columns='win',values=['final','max_dd','avg_exp']).round(3).to_string())
