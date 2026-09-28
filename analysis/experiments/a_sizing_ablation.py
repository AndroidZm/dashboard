"""消融：从基线出发逐个加改动，看收益来自哪里（80 种子，A/B/F），并与“只是多仓位”对比。"""
from a_sizing_lib import *
import time
V={
 '0 baseline 2/8/6 cap.9 pmp15':dict(weights=(0.02,0.08,0.06),cap=0.9,panic_max_pos=15),
 '1 +partial':dict(weights=(0.02,0.08,0.06),cap=0.9,panic_max_pos=15,partial=True),
 '2 +pmpNone':dict(weights=(0.02,0.08,0.06),cap=0.9,panic_max_pos=None),
 '3 +cap1.0':dict(weights=(0.02,0.08,0.06),cap=1.0,panic_max_pos=15),
 '4 +adj15':dict(weights=(0.02,0.15,0.06),cap=0.9,panic_max_pos=15),
 '5 +adj15+partial':dict(weights=(0.02,0.15,0.06),cap=0.9,panic_max_pos=15,partial=True),
 '6 +adj15+partial+cap.95':dict(weights=(0.02,0.15,0.06),cap=0.95,panic_max_pos=15,partial=True),
 '7 +adj15+partial+cap1':dict(weights=(0.02,0.15,0.06),cap=1.0,panic_max_pos=15,partial=True),
 '8 +adj15+partial+cap1+pmpNone':dict(weights=(0.02,0.15,0.06),cap=1.0,panic_max_pos=None,partial=True),
 '9 +partial+cap1+pmpNone (adj8)':dict(weights=(0.02,0.08,0.06),cap=1.0,panic_max_pos=None,partial=True),
 'a adj15 partial cap1 pmp25':dict(weights=(0.02,0.15,0.06),cap=1.0,panic_max_pos=25,partial=True),
 'b adj15 partial cap1 pmp40':dict(weights=(0.02,0.15,0.06),cap=1.0,panic_max_pos=40,partial=True),
 'c adj15 partial cap1 pmpNone mtm':dict(weights=(0.02,0.15,0.06),cap=1.0,panic_max_pos=None,partial=True,sizing='mtm'),
 'd adj12 partial cap1 pmpNone':dict(weights=(0.02,0.12,0.06),cap=1.0,panic_max_pos=None,partial=True),
 'e adj18 partial cap1 pmpNone':dict(weights=(0.02,0.18,0.06),cap=1.0,panic_max_pos=None,partial=True),
 'f adj20 norm0 partial cap1 pmpNone':dict(weights=(0.0,0.20,0.06),cap=1.0,panic_max_pos=None,partial=True),
 'g adj15 partial cap.9 pmpNone':dict(weights=(0.02,0.15,0.06),cap=0.9,panic_max_pos=None,partial=True),
 'h adj15 partial cap.8 pmpNone':dict(weights=(0.02,0.15,0.06),cap=0.8,panic_max_pos=None,partial=True),
 'i adj15 partial cap.85 pmpNone':dict(weights=(0.02,0.15,0.06),cap=0.85,panic_max_pos=None,partial=True),
 'j base partial cap.8 pmpNone':dict(weights=(0.02,0.08,0.06),cap=0.8,panic_max_pos=None,partial=True),
}
names=list(V)
t=time.time()
df=grid([V[n] for n in names],wins=('A','B','F'),nseeds=80)
df['name']=np.repeat(names,3)
df.to_csv('a_sizing_out/ablation.csv',index=False); print(time.time()-t)
df['final']=df.final/1e4
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
p=df.pivot_table(index='name',columns='win',values=['final','cagr','max_dd','dd_p10','avg_exp'])
p[('cagr','A')]*=100; p[('cagr','B')]*=100; p[('cagr','F')]*=100
print(p.round(3).to_string())
