import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    def func(self, x):
        raise NotImplementedError("Func oracle is not implemented.")

    def grad(self, x):
        raise NotImplementedError("Grad oracle is not implemented.")

    def hess(self, x):
        raise NotImplementedError("Hessian oracle is not implemented.")

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError("A should be a symmetric matrix.")
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax = self.matvec_Ax(x)
        m = len(self.b)
        loss = np.sum(np.logaddexp(0, -self.b * Ax)) / m
        reg = 0.5 * self.regcoef * np.linalg.norm(x) ** 2
        return loss + reg

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        m = len(self.b)
        p = expit(-self.b * Ax)
        grad_loss = self.matvec_ATx(p * (-self.b)) / m
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        m = len(self.b)
        p = expit(-self.b * Ax)
        s = p * (1.0 - p)

        hess_loss = self.matmat_ATsA(s) / m

        if scipy.sparse.issparse(hess_loss):
            ident = scipy.sparse.eye(hess_loss.shape[0], format="csr")
            res = hess_loss + self.regcoef * ident
            # Конвертируем в плотный массив для совместимости с np.allclose в тестах
            return res.toarray()
        else:
            return hess_loss + self.regcoef * np.eye(hess_loss.shape[0])


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def create_log_reg_oracle(A, b, regcoef, oracle_type="usual"):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        if scipy.sparse.issparse(A):
            D = scipy.sparse.diags(s)
            return A.T.dot(D.dot(A))
        else:
            return np.dot(A.T * s, A)

    if oracle_type == "usual":
        oracle = LogRegL2Oracle
    elif oracle_type == "optimized":
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError("Unknown oracle_type=%s" % oracle_type)
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    f_x = func(x)
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_eps = np.copy(x)
        x_eps[i] += eps
        grad[i] = (func(x_eps) - f_x) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    f_x = func(x)

    f_plus_i = np.zeros(n)
    for i in range(n):
        x_i = np.copy(x)
        x_i[i] += eps
        f_plus_i[i] = func(x_i)

    for i in range(n):
        for j in range(n):
            x_ij = np.copy(x)
            x_ij[i] += eps
            x_ij[j] += eps
            hess[i, j] = (func(x_ij) - f_plus_i[i] - f_plus_i[j] + f_x) / (eps**2)
    return hess