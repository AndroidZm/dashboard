# 供 b_panic_confirm.py 通过 EXTRA 环境变量载入的最终候选定义
SW5 = dict(swap_on="both", panic_max_pos=None, swap_keep_panic=True, swap_max_n=5)
CANDS = {
    "L1_swap": PConfig(swap_policy="oldest", weights=(0.02, 0.08, 0.04), **SW5),
    "L1_swap_wa12": PConfig(swap_policy="oldest", weights=(0.03, 0.12, 0.04), **SW5),
    "gainabs_w4": PConfig(swap_policy="gain_abs", weights=(0.02, 0.08, 0.04), **SW5),
    "cap1_noswap": PConfig(cap=1.0),
    "L2_cap1_swap": PConfig(cap=1.0, swap_policy="oldest", weights=(0.02, 0.08, 0.04), **SW5),
    "L2_cap1_wa12": PConfig(cap=1.0, swap_policy="oldest", weights=(0.03, 0.12, 0.04), **SW5),
    "L3_cap1_wa16": PConfig(cap=1.0, swap_policy="oldest", weights=(0.03, 0.16, 0.04), **SW5),
}
