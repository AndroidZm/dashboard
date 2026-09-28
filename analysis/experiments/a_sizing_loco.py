"""扩展留一簇：所有 >=8 笔的信号月逐个剔除（80 种子），外加剔除整个 2012 年 / 2024 全年 / 2022 全年。报告相对同 mask 基线的年化差(pp)和回撤。"""
from a_sizing_confirm import CANDS
from a_sizing_lib import *
from multiprocessing import Pool
names=['K0_baseline','K1_base_cap1_part','C1_2_15_6_cap1_part','C1b_2_15_6_cap095_part','C2_0_20_6_cap1_part']
months=['2008-09','2012-01','2012-08','2012-12','2018-02','2018-07','2018-10','2021-02','2022-05','2022-10','2024-02','2026-07']
yrs={'Y2012':[f'2012-{m:02d}' for m in range(1,13)],'Y2022':[f'2022-{m:02d}' for m in range(1,13)],'Y2024':[f'2024-{m:02d}' for m in range(1,13)],
     'top3(2012-01,2022-05,2024-02)':['2012-01','2022-05','2024-02']}
jobs=[]
for n in names:
    for m in months:
        w='A' if m<'2016' else 'B'
        jobs.append((n,w,(m,),m)); jobs.append((n,'F',(m,),m))
    for k,v in yrs.items():
        if k=='Y2012': jobs.append((n,'A',tuple(v),k))
        elif k.startswith('top3'): pass
        else: jobs.append((n,'B',tuple(v),k))
        jobs.append((n,'F',tuple(v),k))
def ev(j):
    n,w,drop,lab=j
    m=run_seeds(D(),cfg_for(CANDS[n],w,drop),seeds=range(80))
    return dict(name=n,win=w,drop=lab,cagr=m['cagr'],final=m['final']/1e4,max_dd=m['max_dd'])
if __name__=='__main__':
    D()
    with Pool(3) as p: rows=p.map(ev,jobs,chunksize=2)
    df=pd.DataFrame(rows); df.to_csv('a_sizing_out/loco.csv',index=False)
    df['k']=df.win+':'+df['drop']
    b=df[df.name=='K0_baseline'].set_index('k')
    df['dC']=(df.cagr-df.k.map(b.cagr))*100; df['dDD']=(df.max_dd-df.k.map(b.max_dd))*100
    pd.set_option('display.width',250); pd.set_option('display.max_columns',60)
    x=df[df.name!='K0_baseline']
    print(x.pivot_table(index='k',columns='name',values='dC').round(2).to_string())
    print(x.pivot_table(index='k',columns='name',values='dDD').round(1).to_string())
    print('min dC per cand:'); print(x.groupby('name').dC.agg(['min','median','max']).round(2))
    print(b[['cagr','max_dd']].round(3).to_string())
