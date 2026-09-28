"""80 种子细网格：wn × wa（wp=0.02/0.06），cap=1.0, partial=True, 不限恐慌持仓。看 K3（A 选出的 0/20/2）附近是平台还是尖峰，两个窗口都看。"""
from a_sizing_lib import *
import time
P=[dict(weights=(a,b,c),cap=1.0,partial=True,panic_max_pos=None) for a in [0,0.01,0.02,0.03] for b in [0.04,0.06,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30] for c in [0.02,0.06]]
t=time.time()
df=grid(P,wins=('A','B','F'),nseeds=80)
df.to_csv('a_sizing_out/plateau.csv',index=False); print(time.time()-t)
df['final']=df.final/1e4
w=df.weights.str.strip('()').str.split(',',expand=True).astype(float); df['wn'],df['wa'],df['wp']=w[0],w[1],w[2]
pd.set_option('display.width',250)
for c in [0.02,0.06]:
    for win in 'ABF':
        s=df[(df.wp==c)&(df.win==win)]
        print(f'--- wp={c} win={win} final'); print(s.pivot_table(index='wn',columns='wa',values='final').round(0).to_string())
        print('maxdd'); print(s.pivot_table(index='wn',columns='wa',values='max_dd').round(3).to_string())
