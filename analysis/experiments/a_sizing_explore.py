"""快速直觉：基线权重整体缩放 k、恐慌档开关。"""
from a_sizing_lib import *
import time
t=time.time()
P=[]
for k in [0.5,1,1.5,2,2.5,3]:
    for cap in [0.9,1.0]:
        for pmp in [15,None]:
            P.append(dict(weights=(0.02*k,0.08*k,0.06*k),cap=cap,panic_max_pos=pmp))
df=grid(P,wins=("A","B"),nseeds=20)
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
df['final']=df.final/1e4
print(df[['weights','cap','panic_max_pos','win','final','cagr','max_dd','avg_exp','n0','n1','n2']].round(3).to_string())
print(time.time()-t)
