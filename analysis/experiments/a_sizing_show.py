import pandas as pd, sys
pd.set_option('display.width',260); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',50)
df=pd.read_csv('a_sizing_out/confirm.csv')
df['key']=df.win+':'+df['drop'].fillna('').replace('','all')
b=df[df.name=='K0_baseline'].set_index('key')
df['d_cagr_pp']=(df.cagr-df.key.map(b.cagr))*100
df['d_dd_pp']=(df.max_dd-df.key.map(b.max_dd))*100
cols=['A:all','A:2012-01','B:all','B:2024-02','B:2018-02','B:2022-05','B:2022-10','F:all','F:2024-02','F:2012-01','F:2018-02','F:2022-05','F:2022-10']
print('delta CAGR pp vs baseline (same mask)'); print(df.pivot_table(index='name',columns='key',values='d_cagr_pp')[cols].round(2).to_string())
print('delta maxDD pp vs baseline (same mask, negative = worse)'); print(df.pivot_table(index='name',columns='key',values='d_dd_pp')[cols].round(1).to_string())
x=df[df['drop'].fillna('')=='']
print(x.pivot_table(index='name',columns='win',values=['final','cagr','max_dd','avg_exp']).round(3).to_string())
