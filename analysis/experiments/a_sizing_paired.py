"""逐种子配对：候选 vs 基线的 CAGR 差（同一种子 = 同一套同日执行顺序随机数），给出中位差、差>0 的比例、各分位。"""
from a_sizing_confirm import CANDS
from a_sizing_lib import *
names=['K1_base_cap1_part','K1b_base_cap095_part','C1_2_15_6_cap1_part','C1b_2_15_6_cap095_part','C2_0_20_6_cap1_part','C3_1_18_6_cap1_part','S10_slots_cap1_part','S30_slots_cap1_part']
rows=[]
for w in 'ABF':
    b=[run(D(),cfg_for(CANDS['K0_baseline'],w),s) for s in range(80)]
    bc=np.array([r['cagr'] for r in b]); bf=np.array([r['final'] for r in b]); bd=np.array([r['max_dd'] for r in b])
    rows.append(dict(name='K0_baseline',win=w,final_med=np.median(bf)/1e4,final_p10=np.percentile(bf,10)/1e4,final_p90=np.percentile(bf,90)/1e4,dd_med=np.median(bd),dd_p10=np.percentile(bd,10)))
    for n in names:
        rr=[run(D(),cfg_for(CANDS[n],w),s) for s in range(80)]
        c=np.array([r['cagr'] for r in rr]); f=np.array([r['final'] for r in rr]); d=np.array([r['max_dd'] for r in rr])
        dc=(c-bc)*100
        rows.append(dict(name=n,win=w,final_med=np.median(f)/1e4,final_p10=np.percentile(f,10)/1e4,final_p90=np.percentile(f,90)/1e4,dd_med=np.median(d),dd_p10=np.percentile(d,10),
                         dcagr_med=np.median(dc),dcagr_p10=np.percentile(dc,10),dcagr_p90=np.percentile(dc,90),frac_pos=(dc>0).mean()))
df=pd.DataFrame(rows); df.to_csv('a_sizing_out/paired.csv',index=False)
pd.set_option('display.width',250)
print(df.round(3).to_string())
