"""执行成本敏感性：次日收盘成交 + 每边 0.5% 滑点（README 的保守口径），候选 vs 基线。"""
from a_sizing_confirm import CANDS
from a_sizing_lib import *
names=['K0_baseline','K1_base_cap1_part','C1_2_15_6_cap1_part','C1b_2_15_6_cap095_part','C2_0_20_6_cap1_part']
rows=[]
for n in names:
    for w in 'ABF':
        m=run_seeds(D(),cfg_for(dict(CANDS[n],lag=1,slip=0.005),w),seeds=range(80))
        rows.append(dict(name=n,win=w,final=m['final']/1e4,cagr=m['cagr']*100,max_dd=m['max_dd']*100))
df=pd.DataFrame(rows); pd.set_option('display.width',200)
print(df.pivot_table(index='name',columns='win',values=['final','cagr','max_dd']).round(2).to_string())
