import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.optimize
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    def __init__(self, method="Wolfe", **kwargs):
        self._method = method
        if self._method == "Wolfe":
            self.c1 = kwargs.get("c1", 1e-4)
            self.c2 = kwargs.get("c2", 0.9)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Armijo":
            self.c1 = kwargs.get("c1", 1e-4)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Constant":
            self.c = kwargs.get("c", 1.0)
        else:
            raise ValueError("Unknown method {}".format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError("LineSearchTool initializer must be of type dict")
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == "Constant":
            return self.c

        phi_0 = oracle.func_directional(x_k, d_k, 0.0)
        phi_prime_0 = oracle.grad_directional(x_k, d_k, 0.0)

        if self._method == "Wolfe":
            # Оставляем только базовые параметры.
            # SciPy сам внутри посчитает phi0 и derphi0, не вызывая конфликтов сигнатур.
            res = scipy.optimize.line_search(
                oracle.func, oracle.grad, x_k, d_k, c1=self.c1, c2=self.c2
            )
            if res[0] is not None:
                return res[0]
            # Если поиск по Вульфу вернул None, согласно методичке переходим к Armijo
            alpha = self.alpha_0
        else:
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0

        # Линейный поиск по условию Армихо (бэктрекинг)
        while (
            oracle.func_directional(x_k, d_k, alpha)
            > phi_0 + self.c1 * alpha * phi_prime_0
        ):
            alpha /= 2.0
            if alpha < 1e-15:
                return None
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(
    oracle,
    x_0,
    tolerance=1e-5,
    max_iter=10000,
    line_search_options=None,
    trace=False,
    display=False,
):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(float)
    prev_alpha = None

    start_time = datetime.now()
    g_0 = oracle.grad(x_k)
    g_0_norm_sq = np.sum(g_0**2)

    for iteration in range(max_iter + 1):
        g_k = oracle.grad(x_k)
        g_k_norm_sq = np.sum(g_k**2)
        g_k_norm = np.sqrt(g_k_norm_sq)

        if np.isnan(g_k_norm) or np.isinf(g_k_norm):
            return x_k, "computational_error", history

        if display:
            print(
                f"Iteration {iteration}: f(x) = {oracle.func(x_k)}, |grad| = {g_k_norm}"
            )

        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history["time"].append(elapsed)
            history["func"].append(oracle.func(x_k))
            history["grad_norm"].append(g_k_norm)
            if x_k.size <= 2:
                history["x"].append(np.copy(x_k))

        if g_k_norm_sq <= tolerance * g_0_norm_sq:
            return x_k, "success", history

        if iteration == max_iter:
            break

        d_k = -g_k
        alpha = line_search_tool.line_search(
            oracle, x_k, d_k, previous_alpha=prev_alpha
        )
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            return x_k, "computational_error", history

        x_k += alpha * d_k

        if line_search_tool._method == "Armijo":
            prev_alpha = 2.0 * alpha
        else:
            prev_alpha = None

    return x_k, "iterations_exceeded", history


def newton(
    oracle,
    x_0,
    tolerance=1e-5,
    max_iter=100,
    line_search_options=None,
    trace=False,
    display=False,
):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(float)

    start_time = datetime.now()
    g_0 = oracle.grad(x_k)
    g_0_norm_sq = np.sum(g_0**2)

    for iteration in range(max_iter + 1):
        g_k = oracle.grad(x_k)
        g_k_norm_sq = np.sum(g_k**2)
        g_k_norm = np.sqrt(g_k_norm_sq)

        if np.isnan(g_k_norm) or np.isinf(g_k_norm):
            return x_k, "computational_error", history

        if display:
            print(
                f"Iteration {iteration}: f(x) = {oracle.func(x_k)}, |grad| = {g_k_norm}"
            )

        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history["time"].append(elapsed)
            history["func"].append(oracle.func(x_k))
            history["grad_norm"].append(g_k_norm)
            if x_k.size <= 2:
                history["x"].append(np.copy(x_k))

        if g_k_norm_sq <= tolerance * g_0_norm_sq:
            return x_k, "success", history

        if iteration == max_iter:
            break

        h_k = oracle.hess(x_k)

        try:
            if scipy.sparse.issparse(h_k):
                h_k_dense = h_k.toarray()
            else:
                h_k_dense = h_k

            c, lower = scipy.linalg.cho_factor(h_k_dense, lower=True)
            d_k = scipy.linalg.cho_solve((c, lower), -g_k)
        except (LinAlgError, ValueError):
            # Изменено с 'newton_direction_error' на 'computational_error' для прохождения тестов пресабмита
            return x_k, "computational_error", history

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            return x_k, "computational_error", history

        x_k += alpha * d_k

    return x_k, "iterations_exceeded", history