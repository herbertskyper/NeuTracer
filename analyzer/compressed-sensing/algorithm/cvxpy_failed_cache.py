import scipy.fftpack as spfft
import cvxpy as cvx
import numpy as np

class CvxpyReconstructor:
    reconstructor_table = {}
    transform_mat_table = {}

    @classmethod
    def get_instance(cls, n, d):
        key = (n, d)
        if key not in cls.reconstructor_table:
            cls.reconstructor_table[key] = cls(n, d)
        return cls.reconstructor_table[key]

    @classmethod
    def get_transform_mat(cls, n, d):
        key = (n, d)
        if key not in cls.transform_mat_table:
            cls.transform_mat_table[key] = np.kron(
                spfft.idct(np.identity(d), norm='ortho', axis=0),
                spfft.idct(np.identity(n), norm='ortho', axis=0)
            )
        return cls.transform_mat_table[key]
    
    # @classmethod
    # def get_instance(cls, n, d):
    #     key = (n, d)
    #     if key not in cls.reconstructor_table:
    #         print(f"[CvxpyReconstructor] Creating new instance for n={n}, d={d}")
    #         cls.reconstructor_table[key] = cls(n, d)
    #     else:
    #         print(f"[CvxpyReconstructor] Using cached instance for n={n}, d={d}")
    #     return cls.reconstructor_table[key]

    # @classmethod
    # def get_transform_mat(cls, n, d):
    #     key = (n, d)
    #     if key not in cls.transform_mat_table:
    #         print(f"[CvxpyReconstructor] Creating new transform_mat for n={n}, d={d}")
    #         cls.transform_mat_table[key] = np.kron(
    #             spfft.idct(np.identity(d), norm='ortho', axis=0),
    #             spfft.idct(np.identity(n), norm='ortho', axis=0)
    #         )
    #     else:
    #         print(f"[CvxpyReconstructor] Using cached transform_mat for n={n}, d={d}")
    #     return cls.transform_mat_table[key]

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.ri = None
        self.transform_mat = self.get_transform_mat(n, d)
        self.vx = cvx.Variable(d * n)
        self.b_param = None
        self.transform_mat_param = None
        self.prob = None

    def set_index(self, index):
        a = index
        b = index
        if self.d > 1:
            for i in range(self.d - 1):
                a = a + self.n * 1
                b = np.concatenate((b, a))
        self.ri = b.astype(int)
        mat = self.transform_mat[self.ri, :]
        self.transform_mat_param = cvx.Parameter(mat.shape)
        self.transform_mat_param.value = mat
        self.b_param = cvx.Parameter((len(self.ri),))
        objective = cvx.Minimize(cvx.norm(self.vx, 1))
        constraints = [self.transform_mat_param @ self.vx == self.b_param]
        self.prob = cvx.Problem(objective, constraints)

    def reconstruct(self, value):
        x = np.zeros((self.n, self.d))
        x[self.ri % self.n, :] = value
        b = x.T.flat[self.ri]
        self.b_param.value = b
        self.prob.solve(solver='SCS', verbose=False)
        x_transformed = np.array(self.vx.value).squeeze()
        x_t = x_transformed.reshape(self.d, self.n).T
        x_re = spfft.idct(spfft.idct(x_t.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
        return x_re

def reconstruct(n, d, index, value):
    reconstructor = CvxpyReconstructor.get_instance(n, d)
    reconstructor.set_index(index)
    return reconstructor.reconstruct(value)