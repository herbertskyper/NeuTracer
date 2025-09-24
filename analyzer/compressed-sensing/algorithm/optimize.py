"""
Sample code automatically generated on 2025-07-18 12:19:40

by geno from www.geno-project.org

from input

parameters
  matrix A
  vector b
variables
  vector x
min
  norm1(x)
st
  A*x == b


Original problem has been transformed into

parameters
  matrix A
  vector b
variables
  vector x
  vector tmp000
min
  sum(tmp000)
st
  A*x-b == vector(0)
  x-tmp000 <= vector(0)
  -(x+tmp000) <= vector(0)


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
from .genosolve import minimize
# from scipy.optimize import minimize
USE_GENO_SOLVER = True
# try:
#     from geno import minimize
#     USE_GENO_SOLVER = True
# except ImportError:
#     from scipy.optimize import minimize
#     USE_GENO_SOLVER = False
#     WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
#           'Run:     pip install genosolver'
#     #print('*' * 63)
#     #print(WRN)
#     #print('*' * 63)



class GenoNLP:
    def __init__(self, A, b, np):
        import cupy
        self.np = cupy
        # print("Using module:", self.np.__name__)
        self.A = cupy.asarray(A, dtype=cupy.float64)
        self.b = cupy.asarray(b, dtype=cupy.float64)
        # assert isinstance(A, self.np.ndarray)
        dim = A.shape
        assert len(dim) == 2
        self.A_rows = dim[0]
        self.A_cols = dim[1]
        # assert isinstance(b, self.np.ndarray)
        dim = b.shape
        assert len(dim) == 1
        self.b_rows = dim[0]
        self.b_cols = 1
        self.x_rows = self.A_cols
        self.x_cols = 1
        self.x_size = self.x_rows * self.x_cols
        self.tmp000_rows = self.A_cols
        self.tmp000_cols = 1
        self.tmp000_size = self.tmp000_rows * self.tmp000_cols
        # the following dim assertions need to hold for this problem
        assert self.A_rows == self.b_rows
        assert self.tmp000_rows == self.x_rows == self.A_cols

    def getLowerBounds(self):
        bounds = []
        bounds += [-inf] * self.x_size
        bounds += [0] * self.tmp000_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [inf] * self.x_size
        bounds += [inf] * self.tmp000_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.xInit = self.np.zeros((self.x_rows, self.x_cols))
        self.tmp000Init = self.np.zeros((self.tmp000_rows, self.tmp000_cols))
        return self.np.hstack((self.xInit.reshape(-1), self.tmp000Init.reshape(-1)))

    def variables(self, _x):
        x = _x[0 : 0 + self.x_size]
        tmp000 = _x[0 + self.x_size : 0 + self.x_size + self.tmp000_size]
        return x, tmp000

    def fAndG(self, _x):
        x, tmp000 = self.variables(_x)
        f_ = self.np.sum(tmp000)
        g_0 = self.np.ones(self.x_rows) * 0
        g_1 = self.np.ones(self.tmp000_rows)
        g_ = self.np.hstack((g_0, g_1))
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        f = ((self.A).dot(x) - self.b)
        return f

    def gradientEqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (self.A)
        g_1 = (self.np.ones((self.A_rows, self.tmp000_rows)) * 0)
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = ((self.A.T).dot(_v))
        gv_1 = (self.np.ones(self.tmp000_rows) * 0)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        f = (x - tmp000)
        return f

    def gradientIneqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (self.np.eye(self.tmp000_rows, self.x_rows))
        g_1 = (-self.np.eye(self.tmp000_rows, self.tmp000_rows))
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint000(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (_v)
        gv_1 = (-_v)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint001(self, _x):
        x, tmp000 = self.variables(_x)
        f = -(x + tmp000)
        return f

    def gradientIneqConstraint001(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (-self.np.eye(self.tmp000_rows, self.x_rows))
        g_1 = (-self.np.eye(self.tmp000_rows, self.tmp000_rows))
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint001(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (-_v)
        gv_1 = (-_v)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

def solve(A, b, np):
    # print("Using module:", np.__name__)
    start = timer()
    NLP = GenoNLP(A, b, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # These are the standard solver options, they can be omitted.
    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-6,
               'max_iter' : 100,
                'max_iter_outer' : 10,
               'm' : 10,
               'ls' : 0,
               'verbose' : 5  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        # check_version('0.1.0')
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jacprod' : NLP.jacProdEqConstraint000},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint000,
                        'jacprod' : NLP.jacProdIneqConstraint000},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint001,
                        'jacprod' : NLP.jacProdIneqConstraint001})
        result = minimize(NLP.fAndG, x0, lb=lb, ub=ub, options=options,
                      constraints=constraints, np=np)
    else:
        # SciPy: for inequality constraints need to change sign f(x) <= 0 -> f(x) >= 0
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jac' : NLP.gradientEqConstraint000},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint000(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint000(x)},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint001(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint001(x)})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          constraints=constraints)

    # assemble solution and map back to original problem
    # x, tmp000 = NLP.variables(result.x)
    if hasattr(result, "x"):
        x_raw = result.x
    else:
        x_raw = result
    x, tmp000 = NLP.variables(x_raw)
    elapsed = timer() - start
    # #print('solving took %.3f sec' % elapsed)
    return result, x, tmp000

def generateRandomData(np):
    np.random.seed(0)
    A = np.random.randn(3, 3)
    b = np.random.randn(3)
    return A, b

if __name__ == '__main__':
    import numpy as np
    import cupy as np  # uncomment this for GPU usage
    #print('\ngenerating random instance')
    print("np module:", np.__name__)
    A, b = generateRandomData(np=np)
    #print('solving ...')
    options = {"verbose": 1}
    result, x, tmp000 = solve(A, b, np=np, options=options)

