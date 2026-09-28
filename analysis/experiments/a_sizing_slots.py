"""等权槽位：单笔 = 权益/N（三档同权），直到 cap。N=5..40 × cap × partial × sizing × panic_max_pos。窗口 A、B、全期。"""
from a_sizing_lib import *
import time
P=[dict(slots=N,cap=cap,partial=pa,sizing=sz,panic_max_pos=None) for N in list(range(5,31))+[33,36,40,50]
   for cap in [0.8,0.9,0.95,1.0] for pa in [False,True] for sz in ['cost','mtm']]
t=time.time()
df=grid(P,wins=("A","B","F"),nseeds=20)
df.to_csv('a_sizing_out/slots.csv',index=False)
print(len(df),time.time()-t)
