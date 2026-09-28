"""阶段 1：三档权重 × cap{0.9,1.0} × panic_max_pos{15,None}，partial=False，sizing=cost。窗口 A、B 各一次（各 1944 组，20 种子）。"""
from a_sizing_lib import *
import time, sys
WN=[0,0.01,0.02,0.03,0.04,0.06]; WA=[0.02,0.03,0.04,0.06,0.08,0.10,0.12,0.15,0.20]; WP=[0,0.01,0.02,0.03,0.04,0.06,0.08,0.10,0.15]
P=[dict(weights=(a,b,c),cap=cap,panic_max_pos=pmp) for a in WN for b in WA for c in WP for cap in [0.9,1.0] for pmp in [15,None]]
for w in sys.argv[1:]:
    t=time.time()
    df=grid(P,wins=(w,),nseeds=20)
    df.to_csv(f'a_sizing_out/grid1_{w}.csv',index=False)
    print(w,len(df),time.time()-t)
