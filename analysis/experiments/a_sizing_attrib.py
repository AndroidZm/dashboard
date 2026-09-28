"""归因：各方案在各信号月簇上投入的资金与盈亏（80 种子平均，单位万），以及回撤发生日期。"""
from a_sizing_lib import *
V={'K0 base':dict(weights=(0.02,0.08,0.06),cap=0.9,panic_max_pos=15),
   'K1 base+cap1+part':dict(weights=(0.02,0.08,0.06),cap=1.0,panic_max_pos=None,partial=True),
   'C1 2/15/6 cap1 part':dict(weights=(0.02,0.15,0.06),cap=1.0,panic_max_pos=None,partial=True),
   'C2 0/20/6 cap1 part':dict(weights=(0.0,0.20,0.06),cap=1.0,panic_max_pos=None,partial=True),
   'S10':dict(slots=10,cap=1.0,panic_max_pos=None,partial=True),
   'S30':dict(slots=30,cap=1.0,panic_max_pos=None,partial=True)}
ym=D().sig.signal_date.str[:7].values
def grp(m):
    if m[:4] in ('2008',): return '2008'
    if m in ('2011-11','2011-12','2012-01'): return '2012-01'
    if m[:4] in ('2012','2013','2011','2014','2015'): return '2012-15other'
    if m in ('2018-01','2018-02'): return '2018-02'
    if m[:4]=='2018' or m[:4]=='2019': return '2018-19other'
    if m in ('2022-04','2022-05'): return '2022-05'
    if m[:4] in ('2022','2023'): return '2022-23other'
    if m in ('2024-01','2024-02'): return '2024-02'
    return m[:4] if m[:4] in ('2016','2017','2020','2021','2025','2026','2024') else m[:4]
G=np.array([grp(m) for m in ym])
pd.set_option('display.width',250)
for win in ['F']:
    rows=[]; dd=[]
    for name,p in V.items():
        cfg=cfg_for(dict(p,record=True),win)
        acc={}
        for sd in range(80):
            r=run(D(),cfg,sd); tr=r['trades']; tr['g']=G[tr.s.values]
            a=tr.groupby('g').agg(amt=('amount','sum'),pnl=('pnl','sum'),n=('s','size'))
            for g,row in a.iterrows():
                x=acc.setdefault(g,np.zeros(3)); x+=row.values/80
            dd.append((name,r['dd_date'],r['max_dd']))
        for g,(amt,pnl,n) in acc.items(): rows.append(dict(name=name,g=g,amt=amt/1e4,pnl=pnl/1e4,n=n))
    df=pd.DataFrame(rows)
    print('window',win,'avg pnl (万) by cluster'); print(df.pivot_table(index='g',columns='name',values='pnl').round(1).to_string())
    print('avg amount invested (万)'); print(df.pivot_table(index='g',columns='name',values='amt').round(1).to_string())
    print('avg n trades'); print(df.pivot_table(index='g',columns='name',values='n').round(1).to_string())
    d=pd.DataFrame(dd,columns=['name','date','dd']); print(d.groupby('name').date.agg(lambda s:s.str[:7].value_counts().head(3).to_dict()))
