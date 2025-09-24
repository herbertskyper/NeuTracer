import scipy.fftpack as spfft
import cvxpy as cvx
import numpy as np
from time import perf_counter
from scipy.optimize import linprog
import time
import torch
import cupy as cp
from cupy.cuda import cusolver
from cupy.cuda import device
# import osqp
# from .custom_solver import myOSQP

def compressed_sensing_direct(transform_mat, b):
    """
    直接展开法求解 min ‖x‖₁ s.t. transform_mat @ x == b
    完全替代 CVXPY 的解决方案
    """
    m, n = transform_mat.shape
    
    # 1. 构造线性规划参数
    # 目标函数: min [0,..0, 1,..1] · [x; t]
    # transform_mat = transform_mat.astype(np.float32)
    # b = b.astype(np.float32)

    # # 构造线性规划参数时使用 float32
    # c = np.hstack([np.zeros(n), np.ones(n)]).astype(np.float32)
    # A_eq = np.hstack([transform_mat, np.zeros((m, n))]).astype(np.float32)
    # A_ub = np.vstack([
    #     np.hstack([np.eye(n), -np.eye(n)]),
    #     np.hstack([-np.eye(n), -np.eye(n)])
    # ]).astype(np.float32)
    # b_ub = np.zeros(2 * n).astype(np.float32)
    c = np.hstack([np.zeros(n), np.ones(n)])
    
    # 等式约束: transform_mat @ x = b → [transform_mat | 0] [x; t] = b
    A_eq = np.hstack([transform_mat, np.zeros((m, n))])
    
    # 不等式约束: -t ≤ x ≤ t → 
    A_ub = np.vstack([
        np.hstack([ np.eye(n), -np.eye(n)]),   # x_i - t_i ≤ 0
        np.hstack([-np.eye(n), -np.eye(n)])    # -x_i - t_i ≤ 0
    ])
    b_ub = np.zeros(2 * n)
    # 2. 求解线性规划 - 使用高效HiGHS求解器
    start_time = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b,
                  method='highs', bounds=(None, None),options={"disp": False})
    
    # 3. 提取解
    if res.success:
        x_sol = res.x[:n]
        solve_time = time.time() - start_time
        return x_sol, solve_time
    else:
        raise RuntimeError("求解失败: " + res.message)

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho',
                     axis=0)


def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho',
                      axis=0)


def reconstruct(n, d, index, value):
    """
    压缩感知采样重建算法
    :param n: 需重建数据的数据量
    :param d: 需重建数据的维度
    :param index: 采样点的时间维度坐标 属于[0, n-1]
    :param value: 采样点的KPI值，shape=(m, d), m为采样数据量
    :return:x_re: 重建的KPI数据，shape=(n, d)
    """
    x = np.zeros((n, d))
    x[index, :] = value

    a = index
    b = index

    if d > 1:
        for i in range(d - 1):
            a = a + n * 1
            b = np.concatenate((b, a))
    index = b.astype(int)

    # random sample of indices

    ri = index
    b = x.T.flat[ri]
    # b = value.T.flat
    # b = np.expand_dims(b, axis=1)

    # create dct matrix operator using kron (memory errors for large ny*nx)
    transform_mat = np.kron(
        spfft.idct(np.identity(d), norm='ortho', axis=0),
        spfft.idct(np.identity(n), norm='ortho', axis=0)
    )
    # print(transform_mat)
    transform_mat = transform_mat[ri, :]  # same as phi times kron
    # print(f'Transform matrix shape: {transform_mat.shape}')
    # print(f'vx shape: {d} {n}')
    # print(f'b shape: {b.shape}')
    # print(transform_mat)
    x_transformed, solve_time = compressed_sensing_direct(transform_mat, b)
    # print(solve_time)

    # # do L1 optimization
    # vx = cvx.Variable(d * n)
    # objective = cvx.Minimize(cvx.norm(vx, 1))
    # constraints = [transform_mat @ vx == b]
    # prob = cvx.Problem(objective, constraints)
    # prob.solve(solver='SCS', verbose = False) #13
    # print(f'SCS solve time: {perf_counter()} seconds')
    # prob.solve(solver='CLARABEL') #20
    # prob.solve(solver='OSQP',  backend = ) # 


    # vx = cvx.Variable(d * n)

    # b_param = cvx.Parameter(b.shape)
    # transform_mat_param = cvx.Parameter(transform_mat.shape)
    # objective = cvx.Minimize(cvx.norm(vx, 1))
    # constraints = [transform_mat_param @ vx == b_param]

    # prob = cvx.Problem(objective, constraints)

    # b_param.value = b
    # transform_mat_param.value = transform_mat
    # prob.solve(solver='SCS', verbose=True)

    # x_transformed = np.array(vx.value).squeeze()
    # reconstruct signal
    x_t = x_transformed.reshape(d, n).T  # stack columns
    x_re = idct2(x_t)

    # confirm solution
    if not np.allclose(b, x_re.T.flat[ri]):
        print('Warning: values at sample indices don\'t match original.')

    return x_re


    # print(f'Transform matrix shape: {transform_mat.shape}')
    # print(f'vx shape: {d} {n}')
    # print(f'b shape: {b.shape}')
    # sin_freq = 1
    # sin_amp = 1.0
    # sin_signal = sin_amp * np.sin(2 * np.pi * sin_freq * np.arange(d * n))
    # sample_sin = sin_signal[ri]
    # combine = b + sample_sin

    # # do L1 optimization

    # vx = cvx.Variable(d * n)
    # objective = cvx.Minimize(cvx.norm(vx, 1))
    # constraints = [transform_mat @ vx == combine]
    # prob = cvx.Problem(objective, constraints)
    # # prob.solve(solver='SCS', use_indirect=True, gpu=True)
    # prob.solve(solver='OSQP')
    # x_transformed = np.array(vx.value).squeeze()
    # x_filtered = x_transformed - sin_signal
    # x_t = x_filtered.reshape(d, n).T  # stack columns
    # # x_t = x_transformed.reshape(d, n).T  # stack columns
    # x_re = idct2(x_t)
    # # print(f'{b}')
    # # print(f'{x_re.T.flat[ri]}')

    # # confirm solution
    # if not np.allclose(b, x_re.T.flat[ri], rtol=1e-2, atol=1e-3):
    #     print('Warning: values at sample indices don\'t match original.')

    # return x_re




    # from .optimize import minimize

    # def fg(x):
    #     Ax_minus_b = transform_mat @ x - b
    #     f = np.sum(Ax_minus_b**2) + lmbd * np.sum(np.abs(x))
    #     g = 2 * transform_mat.T @ Ax_minus_b + lmbd * np.sign(x)
    #     return f, g

    # x0 = np.zeros(transform_mat.shape[1])
    # lmbd = 1e-3  # 可调
    # res = minimize(fg, x0)
    # x_transformed = res.x



    # def l1_minimization_gpu(A, b, lmbd=1e-3, max_iter=1000, tol=1e-6):
    #     import torch
    #     """
    #     用PyTorch在GPU上做L1最小化: min_x ||Ax-b||^2 + lambda*||x||_1
    #     """
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     A = torch.tensor(A, dtype=torch.float32, device=device)
    #     b = torch.tensor(b, dtype=torch.float32, device=device)
    #     x = torch.zeros(A.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    #     optimizer = torch.optim.Adam([x], lr=1e-2)
    #     for i in range(max_iter):
    #         optimizer.zero_grad()
    #         loss = torch.norm(A @ x - b, 2)**2 + lmbd * torch.norm(x, 1)
    #         loss.backward()
    #         optimizer.step()
    #         if loss.item() < tol:
    #             break
    #     return x.detach().cpu().numpy()
    # x_transformed = l1_minimization_gpu(transform_mat, b)
# -------------------------------------------------------------------------
    # from .optimize import solve
    # # options = {"verbose": 0}
    # import cupy 
    # result, x_opt, tmp = solve(transform_mat, b, np = cupy)
    # x_transformed = x_opt

    # x_t = x_transformed.reshape(d, n).T  # stack columns
    # x_re = idct2(x_t.get())
# -------------------------------------------------------------------------
    # x_t = x_transformed.reshape(d, n).T  # stack columns
    # x_re = idct2(x_t)
    # # print(f'{b}')
    # # print(f'{x_re.T.flat[ri]}')

    # # # confirm solution
    # # if not np.allclose(b, x_re.T.flat[ri], rtol=1e-3, atol=1e-3):
    # #     print('Warning: values at sample indices don\'t match original.')

    # return x_re
