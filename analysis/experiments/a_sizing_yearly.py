"""逐年收益（80 种子中位，全期 2006-2026），看增益在哪些年份、是否只是仓位。"""
from a_sizing_confirm import CANDS
from a_sizing_lib import *
names=['K0_baseline','K1_base_cap1_part','C1_2_15_6_cap1_part','C2_0_20_6_cap1_part']
out={}
expo={}
for n in names:
    ys=[];es=[]
    for sd in range(80):
        r=run(D(),cfg_for(dict(CANDS[n],record=True),'F'),sd)
        eq=r['equity']; eq.index=pd.to_datetime(eq.index)
        y=eq.groupby(eq.index.year).last(); y0=pd.concat([pd.Series([600000.0],index=[2005]),y]); ys.append(y0.pct_change().dropna())
        ex=r['exposure']; ex.index=pd.to_datetime(ex.index); es.append(ex.groupby(ex.index.year).mean())
    out[n]=pd.concat(ys,axis=1).median(axis=1)*100; expo[n]=pd.concat(es,axis=1).median(axis=1)*100
pd.set_option('display.width',200)
print('yearly return % (median of 80 seeds)'); print(pd.DataFrame(out).round(1).to_string())
print('yearly avg exposure %'); print(pd.DataFrame(expo).round(0).to_string())
