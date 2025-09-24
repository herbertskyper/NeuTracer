import numpy as np
import scipy.fftpack as spfft

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# 生成测试矩阵
np.random.seed(42)
x = np.random.randn(8, 3)
np.savetxt("test_idct2_input.csv", x, delimiter=",")

# 计算idct2
y = idct2(x)
np.savetxt("test_idct2_py.csv", y, delimiter=",")