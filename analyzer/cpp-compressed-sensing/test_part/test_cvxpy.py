import numpy as np

n, d = 10, 1
np.random.seed(42)
x_true = np.random.randn(n, d)
sample_rate = 0.5
m = int(n * sample_rate)
index = np.sort(np.random.choice(n, m, replace=False))
value = x_true[index, :]

np.savetxt("test_index_cal.txt", index, fmt="%d")
np.savetxt("test_value_cal.txt", value, delimiter=",")
np.savetxt("test_x_true_cal.txt", x_true, delimiter=",")

from mycvxpy import reconstruct  # 假设你的reconstruct函数已在PYTHONPATH

x_re = reconstruct(n, d, index, value)
np.savetxt("test_x_re_cal_py.txt", x_re, delimiter=",")