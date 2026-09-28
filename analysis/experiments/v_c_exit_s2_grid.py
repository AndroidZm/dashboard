"""v_c_exit 第 2 步：20 种子网格（A/B/F）——单段移动止盈 act x trail x sl、两段式、固定止盈 x 止损。用于双向样本外选择与邻域检查。"""
import sys, time, itertools
sys.path.insert(0, '/home/user/dashboard/analysis/experiments')
from v_c_exit_lib import *
if __name__ == '__main__':
    t = time.time()
    RU = {'base': CANDS['base']}
    SLS = [0.4, 0.5, 0.6, 0.7, 0.8, None]
    for a, w, sl in itertools.product([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0], [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35], SLS):
        RU[f'tr|{a}|{w}|{sl}'] = dict(kind='trail', act=a, trail=w, sl=sl, tp=None)
    for tp, sl in itertools.product([0.3, 0.4, 0.5, 0.6, 0.8, 1.0], SLS):
        RU[f'fx|{tp}|{sl}'] = dict(kind='fixed', tp=tp, sl=sl)
    for a, w, a2, w2, sl in itertools.product([0.4, 0.5, 0.6], [0.05, 0.075, 0.1], [0.8, 1.0, 1.5], [0.2, 0.25, 0.3, 0.35], [0.4, 0.6, 0.7, None]):
        RU[f't2|{a}|{w}|{a2}|{w2}|{sl}'] = dict(kind='trail2', act=a, trail=w, act2=a2, trail2=w2, sl=sl, tp=None)
    print('rules', len(RU))
    jobs = [(n, r, win, (), 20, {}, 0) for n, r in RU.items() for win in 'ABF']
    df = evaluate(jobs, procs=3)
    df.to_csv('/home/user/dashboard/analysis/experiments/v_c_exit_out/s2_grid20.csv', index=False)
    print('elapsed', time.time() - t)
