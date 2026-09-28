"""v_c_exit 第 4 步（80 种子）：候选 ±1 步邻域、去 2024-02 与留一簇、执行压力（lag=1 + 0.5% 滑点，另加出场也次日）、终点敏感性。"""
import sys, time
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_c_exit_lib import *

def T(a, w, sl, tp=None):
    return dict(kind='trail', act=a, trail=w, sl=sl, tp=tp)
def T2(a, w, a2, w2, sl):
    return dict(kind='trail2', act=a, trail=w, act2=a2, trail2=w2, sl=sl, tp=None)

NB = {  # 邻域：每个参数 ±1 步
    'C1': T(0.5, 0.1, 0.7), 'C1_a45': T(0.45, 0.1, 0.7), 'C1_a55': T(0.55, 0.1, 0.7), 'C1_a40': T(0.4, 0.1, 0.7), 'C1_a60': T(0.6, 0.1, 0.7),
    'C1_t075': T(0.5, 0.075, 0.7), 'C1_t125': T(0.5, 0.125, 0.7), 'C1_sl60': T(0.5, 0.1, 0.6), 'C1_slNone': T(0.5, 0.1, None), 'C1_sl40': T(0.5, 0.1, 0.4),
    'C1_a45_t075': T(0.45, 0.075, 0.7), 'C1_a55_t125': T(0.55, 0.125, 0.7), 'C1_a45_t125': T(0.45, 0.125, 0.7), 'C1_a55_t075': T(0.55, 0.075, 0.7),
    'C2': T(0.45, 0.05, 0.4), 'C2_a40': T(0.4, 0.05, 0.4), 'C2_a50': T(0.5, 0.05, 0.4), 'C2_t025': T(0.45, 0.025, 0.4), 'C2_t075': T(0.45, 0.075, 0.4),
    'C2_sl35': T(0.45, 0.05, 0.35), 'C2_sl45': T(0.45, 0.05, 0.45), 'C2_sl50': T(0.45, 0.05, 0.5),
    'C3': T2(0.5, 0.075, 1.0, 0.3, 0.7), 'C3_a45': T2(0.45, 0.075, 1.0, 0.3, 0.7), 'C3_a55': T2(0.55, 0.075, 1.0, 0.3, 0.7),
    'C3_t05': T2(0.5, 0.05, 1.0, 0.3, 0.7), 'C3_t10': T2(0.5, 0.1, 1.0, 0.3, 0.7), 'C3_x80': T2(0.5, 0.075, 0.8, 0.3, 0.7),
    'C3_x120': T2(0.5, 0.075, 1.2, 0.3, 0.7), 'C3_w25': T2(0.5, 0.075, 1.0, 0.25, 0.7), 'C3_w35': T2(0.5, 0.075, 1.0, 0.35, 0.7),
    'C3_sl60': T2(0.5, 0.075, 1.0, 0.3, 0.6), 'C3_slNone': T2(0.5, 0.075, 1.0, 0.3, None), 'C3_sl40': T2(0.5, 0.075, 1.0, 0.3, 0.4),
    'C4': T(1.0, 0.3, 0.7), 'C4_a80': T(0.8, 0.3, 0.7), 'C4_a120': T(1.2, 0.3, 0.7), 'C4_t25': T(1.0, 0.25, 0.7), 'C4_t35': T(1.0, 0.35, 0.7),
    'C4_sl60': T(1.0, 0.3, 0.6), 'C4_slNone': T(1.0, 0.3, None), 'C4_sl40': T(1.0, 0.3, 0.4),
    'fx_tp40_sl70': dict(kind='fixed', tp=0.4, sl=0.7), 'fx_tp50_sl70': dict(kind='fixed', tp=0.5, sl=0.7),
}
MAIN = ['base', 'C1', 'C2', 'C3', 'C4']
ENDS_B = ['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31', '2025-06-30', '2025-12-31']
ENDS_A = ['2008-12-31', '2010-12-31', '2012-12-31', '2014-12-31', '2015-06-12', '2015-12-31']

if __name__ == '__main__':
    t = time.time()
    RU = {'base': CANDS['base'], **NB}
    jobs = []
    for n, r in RU.items():
        for w in 'ABF':
            jobs.append((n, r, w, (), 80, {}, 0))
    for n in MAIN:
        r = RU[n]
        # 去簇
        for m in LOCO:
            for w in (['A', 'F'] if m < '2016' else ['B', 'F']):
                jobs.append((n, r, w, (m,), 80, {}, 0))
        for w in 'ABF':
            jobs.append((n, r, w, tuple(LOCO), 80, {}, 0))
            # 执行压力
            jobs.append((n, r, w, (), 80, {'lag': 1, 'slip': 0.005}, 0))
            jobs.append((n, r, w, (), 80, {'lag': 1, 'slip': 0.005}, 1))
    df = evaluate(jobs, procs=3)
    df.to_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s4_confirm80.csv', index=False)
    # 终点敏感性（单独跑，end 不同）
    import v_c_exit_lib as L
    D = getD()
    rows = []
    for n in MAIN:
        for st, ends in (('2016-01-01', ENDS_B), ('2006-01-04', ENDS_A + ENDS_B)):
            for en in ends:
                cfg = Config(start=st, end=en)
                m = seeds_stats(D, RU[n], cfg)
                rows.append(dict(name=n, start=st, end=en, final=m['final'], cagr=m['cagr'], dd=m['dd']))
    pd.DataFrame(rows).to_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s4_ends.csv', index=False)
    print('elapsed', time.time() - t)
