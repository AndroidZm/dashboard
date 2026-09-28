"""分档仓位上限：平常/调整档只能用到 cap_np，恐慌档可用到 1.0（给恐慌簇留现金）。partial 语义：放不下就买剩余额度。
20 种子网格，窗口 A/B/F。"""
from a_sizing_lib import *
from functools import partial as fpartial
from multiprocessing import Pool
import time

def wf_tiercap(s, t, st, w, caps):
    k = st["tier"]
    room = caps[k] * st["equity"] - st["exposure"]
    if room <= 0:
        return 0.0
    return min(w[k], room / st["equity"])

P=[]
for wn in [0.01,0.02]:
    for wa in [0.04,0.08,0.15]:
        for wp in [0.02,0.04,0.06,0.10]:
            for cnp in [0.5,0.6,0.7,0.8,0.9,1.0]:
                P.append(((wn,wa,wp),cnp))
def ev(j):
    (w,cnp),win,ns=j
    cfg=cfg_for(dict(weight_fn=fpartial(wf_tiercap,w=w,caps=(cnp,cnp,1.0)),cap=1.0,partial=True,panic_max_pos=None),win)
    m=run_seeds(D(),cfg,seeds=range(ns))
    return dict(w=str(w),cnp=cnp,win=win,final=m['final']/1e4,cagr=m['cagr'],max_dd=m['max_dd'],avg_exp=m['avg_exposure'],n0=m['n0'],n1=m['n1'],n2=m['n2'])
if __name__=='__main__':
    D(); t=time.time()
    with Pool(3) as p: rows=p.map(ev,[(x,w,20) for x in P for w in 'ABF'],chunksize=4)
    df=pd.DataFrame(rows); df.to_csv('a_sizing_out/tiercap.csv',index=False); print(time.time()-t)
    pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
    print(df.pivot_table(index=['w'],columns=['win','cnp'],values='final').round(0).to_string())
    print(df.pivot_table(index=['w'],columns=['win','cnp'],values='max_dd').round(2).to_string())
    print(df.pivot_table(index=['w'],columns=['win','cnp'],values='n2').round(0).to_string())
