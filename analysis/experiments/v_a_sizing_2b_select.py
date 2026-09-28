"""检查 2 分析：跨窗口选择 + 邻域表。"""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
df = pd.read_csv('v_a_sizing_out/grid.csv')
W = df.pivot_table(index='key', columns='win', values=['final', 'cagr', 'dd'])
base = W.loc[[k for k in W.index if 'pmp15' in k][0]]
print('baseline', base.round(2).to_dict())
fam = W[[('part1' in k) for k in W.index]]
print('n configs partial family', len(fam), 'spearman A-B final:', round(fam[('final', 'A')].corr(fam[('final', 'B')], method='spearman'), 3))
fam1 = fam[[('cap1.0' in k) for k in fam.index]]
print('cap1 only n', len(fam1), 'spearman', round(fam1[('final', 'A')].corr(fam1[('final', 'B')], method='spearman'), 3))
for src, dst in [('A', 'B'), ('B', 'A')]:
    for lim in [-25, -27.5, -30, -31, -32, -33, -35, -40, -100]:
        d = fam[fam[('dd', src)] >= lim].sort_values(('final', src), ascending=False)
        if not len(d): continue
        t = d.iloc[0]; k = d.index[0]
        t5 = d.head(5)
        print(f'[{src}->{dst}] dd>={lim:5.1f}: {k:38s} {src} {t[("final",src)]:6.1f}/{t[("dd",src)]:5.1f} -> {dst} {t[("final",dst)]:6.1f}/{t[("dd",dst)]:5.1f} '
              f'(dCAGR vs base {t[("cagr",dst)]-base[("cagr",dst)]:+.2f}pp) | top5 {dst} med {t5[("final",dst)].median():.1f} [{t5[("final",dst)].min():.1f},{t5[("final",dst)].max():.1f}]')
# 邻域
for c in [1.0, 0.95, 0.9]:
    for w in 'BAF':
        s = df[(df.partial) & (df.cap == c) & (df.win == w)]
        print(f'--- cap {c} window {w}: final (万)')
        print(s.pivot_table(index='wn', columns='wa', values='final').round(1).to_string())
        print('dd'); print(s.pivot_table(index='wn', columns='wa', values='dd').round(1).to_string())
for w in 'BAF':
    s = df[(~df.partial) & (df.win == w) & (df.pmp.isna())]
    print(f'--- partial=False wn=0.02 window {w}'); print(s.pivot_table(index='cap', columns='wa', values='final').round(1).to_string())
