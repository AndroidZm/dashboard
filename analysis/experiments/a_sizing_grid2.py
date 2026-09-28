"""阶段 2：cap=1.0 下 partial × sizing × panic_max_pos{25,None}，缩小的权重网格；另跑 cap{0.8,0.95} 做前沿。窗口 A、B。"""
from a_sizing_lib import *
import time, sys
WN=[0,0.01,0.02,0.03,0.04,0.06]; WA=[0.03,0.04,0.06,0.08,0.10,0.12,0.15,0.20]; WP=[0,0.02,0.04,0.08]
W=[(a,b,c) for a in WN for b in WA for c in WP]
P=[dict(weights=w,cap=1.0,panic_max_pos=pmp,partial=pa,sizing=sz) for w in W for pmp in [25,None] for pa in [False,True] for sz in ['cost','mtm']]
P2=[dict(weights=w,cap=cap,panic_max_pos=None,partial=pa) for w in W for cap in [0.8,0.95] for pa in [False,True]]
for w in sys.argv[1:]:
    t=time.time()
    df=grid(P,wins=(w,),nseeds=20); df.to_csv(f'a_sizing_out/grid2_{w}.csv',index=False)
    df=grid(P2,wins=(w,),nseeds=20); df.to_csv(f'a_sizing_out/grid2cap_{w}.csv',index=False)
    print(w,time.time()-t)
