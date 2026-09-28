"""80 种子确认 + 剔簇检验：窗口 A/B/F × {全部, 剔 2024-02, 逐个剔大簇}，与同 mask 下的基线对比。"""
from a_sizing_lib import *
from a_sizing_adaptive import wf_n30
from functools import partial as fpartial
import time, sys, json

BASE=dict(weights=(0.02,0.08,0.06),cap=0.9,panic_max_pos=15)
CANDS={
 'K0_baseline':BASE,
 'K1_base_cap1_part':dict(weights=(0.02,0.08,0.06),cap=1.0,panic_max_pos=None,partial=True),
 'K1b_base_cap095_part':dict(weights=(0.02,0.08,0.06),cap=0.95,panic_max_pos=None,partial=True),
 'K1c_base_cap1_pmp15':dict(weights=(0.02,0.08,0.06),cap=1.0,panic_max_pos=15,partial=False),
 'K3_Asel_0_20_2':dict(weights=(0.0,0.20,0.02),cap=1.0,panic_max_pos=None,partial=True),
 'K4_Bsel_2_6_8':dict(weights=(0.02,0.06,0.08),cap=1.0,panic_max_pos=None,partial=True),
 'K5_Bsel_3_4_15':dict(weights=(0.03,0.04,0.15),cap=1.0,panic_max_pos=None,partial=False),
 'K6_n30_10_03':dict(weight_fn=fpartial(wf_n30,N0=10,b=0.3),cap=1.0,panic_max_pos=None,partial=True),
}
CANDS.update({
 'C1_2_15_6_cap1_part':dict(weights=(0.02,0.15,0.06),cap=1.0,panic_max_pos=None,partial=True),
 'C1b_2_15_6_cap095_part':dict(weights=(0.02,0.15,0.06),cap=0.95,panic_max_pos=None,partial=True),
 'C2_0_20_6_cap1_part':dict(weights=(0.0,0.20,0.06),cap=1.0,panic_max_pos=None,partial=True),
 'C3_1_18_6_cap1_part':dict(weights=(0.01,0.18,0.06),cap=1.0,panic_max_pos=None,partial=True),
})
for N in [10,12,15,20,25,30,36]:
    CANDS[f'S{N}_slots_cap1_part']=dict(slots=N,cap=1.0,panic_max_pos=None,partial=True)

MASKS=[('A',()),('B',()),('F',()),('B',('2024-02',)),('F',('2024-02',))]+[('A',('2012-01',)),('F',('2012-01',))]+\
      [(w,(m,)) for m in ['2018-02','2022-05','2022-10'] for w in ['B','F']]

def _ev(args):
    name,w,drop=args
    m=run_seeds(D(),cfg_for(CANDS[name],w,drop),seeds=range(80))
    return dict(name=name,win=w,drop='|'.join(drop),final=m['final']/1e4,cagr=m['cagr'],max_dd=m['max_dd'],dd_p10=m['dd_p10'],
                cagr_p10=m['cagr_p10'],cagr_p90=m['cagr_p90'],avg_exp=m['avg_exposure'],n=m['n_trades'],n0=m['n0'],n1=m['n1'],n2=m['n2'])

if __name__=='__main__':
    from multiprocessing import Pool
    names=sys.argv[1:] or list(CANDS)
    if 'K0_baseline' not in names: names=['K0_baseline']+names
    D(); t=time.time()
    jobs=[(n,w,d) for n in names for w,d in MASKS]
    with Pool(3) as pool: rows=pool.map(_ev,jobs,chunksize=1)
    df=pd.DataFrame(rows); out='a_sizing_out/confirm.csv'
    try:
        old=pd.read_csv(out); old=old[~old.name.isin(names)]; df=pd.concat([old,df],ignore_index=True)
    except FileNotFoundError: pass
    df.to_csv(out,index=False); print(time.time()-t)
    pd.set_option('display.width',260); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',50)
    df['key']=df.win+':'+df['drop'].fillna('').replace('','all')
    b=df[df.name=='K0_baseline'].set_index('key')
    df['d_cagr_pp']=(df.cagr-df.key.map(b.cagr))*100
    df['d_dd_pp']=(df.max_dd-df.key.map(b.max_dd))*100
    print(df.pivot_table(index='name',columns='key',values='d_cagr_pp').round(2).to_string())
    print(df.pivot_table(index='name',columns='key',values='d_dd_pp').round(1).to_string())
    print(df[df['drop'].fillna('')==''].pivot_table(index='name',columns='win',values=['final','cagr','max_dd','avg_exp']).round(3).to_string())
