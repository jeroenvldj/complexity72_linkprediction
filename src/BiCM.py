# -*- coding: utf-8 -*-s
"""
Created on Fri Feb 16 10:41:04 2018

@author: lePiddu
"""

import numpy as np
from numba import jit
from scipy import optimize as opt

@jit()
def vec2mat(x, y):
    return np.atleast_2d(x).T @ np.atleast_2d(y)

@jit
def probability_matrix(x, y):
    return vec2mat(x, y) / (1 + vec2mat(x, y))

@jit()
def sum_combination(x, y):
    """
    This function computes a matrix in which each element is the exponential
    of the sum of the corresponding elements of x and y
    """
    return np.exp(np.atleast_2d(x).T + np.atleast_2d(y))

@jit()
def eqs(xx, d_rows, d_cols, multiplier_rows, multiplier_cols, nrows, out_res):
    xixa = vec2mat(xx[:nrows], xx[nrows:])
    xixa /= 1 + xixa
    out_res[:nrows] = (xixa * multiplier_cols).sum(axis=1) - d_rows
    out_res[nrows:] = (xixa.T * multiplier_rows).sum(axis=1) - d_cols

@jit()
def jac(xx, multiplier_rows, multiplier_cols, nrows, ncols, out_J_T):
    xixa_tilde = 1 / (1 + vec2mat(xx[:nrows], xx[nrows:]))
    xixa_tilde *= xixa_tilde
    lower_block_T = xixa_tilde * xx[nrows:]
    up_left_block = (lower_block_T * multiplier_cols).sum(axis=1) * \
                        np.eye(nrows)
    upper_block_T = xixa_tilde.T * xx[:nrows]
    lo_right_block = (upper_block_T * multiplier_rows).sum(axis=1) * \
                        np.eye(ncols)
    out_J_T[:nrows, :nrows] = up_left_block
    out_J_T[nrows:, nrows:] = lo_right_block
    out_J_T[nrows:, :nrows] = (upper_block_T.T * multiplier_cols).T
    out_J_T[:nrows, nrows:] = (lower_block_T.T * multiplier_rows).T

@jit()
def eqs_W(xx, d_rows, d_cols, multiplier_rows, multiplier_cols, nrows, out_res):
    xixa = vec2mat(xx[:nrows], xx[nrows:])
    xixa /= 1 - xixa
    out_res[:nrows] = (xixa * multiplier_cols).sum(axis=1) - d_rows
    out_res[nrows:] = (xixa.T * multiplier_rows).sum(axis=1) - d_cols

@jit()
def jac_W(xx, multiplier_rows, multiplier_cols, nrows, ncols, out_J_T):
    xixa_tilde = 1 / (1 - vec2mat(xx[:nrows], xx[nrows:]))
    xixa_tilde *= xixa_tilde
    lower_block_T = xixa_tilde * xx[nrows:]
    up_left_block = (lower_block_T * multiplier_cols).sum(axis=1) * \
                        np.eye(nrows)
    upper_block_T = xixa_tilde.T * xx[:nrows]
    lo_right_block = (upper_block_T * multiplier_rows).sum(axis=1) * \
                        np.eye(ncols)
    out_J_T[:nrows, :nrows] = up_left_block
    out_J_T[nrows:, nrows:] = lo_right_block
    out_J_T[nrows:, :nrows] = (upper_block_T.T * multiplier_cols).T
    out_J_T[:nrows, nrows:] = (lower_block_T.T * multiplier_rows).T

@jit()
def reform_eqs_W(xx, d_rows, d_cols, multiplier_rows, multiplier_cols, nrows, out_res):
    """
    Reform equations are written by removing the numerator. I believe it
    is going to be a bit more stable, especially looking at the equations and
    the jacobian
    """
    xixa = 1 / (sum_combination(xx[:nrows], xx[nrows:]) - 1)
    out_res[:nrows] = (xixa * multiplier_cols).sum(axis=1) - d_rows
    out_res[nrows:] = (xixa.T * multiplier_rows).sum(axis=1) - d_cols


@jit()
def reform_jac_W(xx, multiplier_rows, multiplier_cols, nrows, ncols, out_J_T):
    xixa = sum_combination(xx[:nrows], xx[nrows:])
    xixa_tilde = - xixa / np.power((xixa - 1), 2)
    up_left_block = (xixa_tilde * multiplier_cols).sum(axis=1) * \
                        np.eye(nrows)
    lo_right_block = (xixa_tilde.T * multiplier_rows).sum(axis=1) * \
                        np.eye(ncols)
    out_J_T[:nrows, :nrows] = up_left_block
    out_J_T[nrows:, nrows:] = lo_right_block
    out_J_T[nrows:, :nrows] = (xixa_tilde * multiplier_cols).T
    out_J_T[:nrows, nrows:] = (xixa_tilde.T * multiplier_rows).T


class bicm_leastSquares:
    def __init__(self, degree_sequence, n_rows, n_cols):
        ##### Full problem parameters
        self.dseq_rows = np.copy(degree_sequence[:n_rows])
        self.dseq_cols = np.copy(degree_sequence[n_rows:])
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.dim = self.n_rows + self.n_cols
        self.n_edges = np.sum(self.dseq_rows)
        self.x = None
        self.y = None
        self.xy = None
        ##### Reduced problem parameters
        self.is_reduced = False
        self.r_dseq_rows = None
        self.r_dseq_cols = None
        self.r_n_rows = None
        self.r_n_cols = None
        self.r_invert_dseq_rows = None
        self.r_invert_dseq_cols = None
        self.r_dim = None
        self.rows_multiplicity = None
        self.cols_multiplicity = None
        #### Problem solutions
        self.J_T = None
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        self.p_adjacency = None
        #### Problem (reduced) residuals
        self.residuals = None
        self.final_result = None

    def degree_degeneration(self):
        self.r_dseq_rows, self.r_invert_dseq_rows, self.rows_multiplicity \
            = np.unique(self.dseq_rows, 0, 1, 1)
        self.r_dseq_cols, self.r_invert_dseq_cols, self.cols_multiplicity \
            = np.unique(self.dseq_cols, 0, 1, 1)
        self.r_n_rows = self.r_dseq_rows.size
        self.r_n_cols = self.r_dseq_cols.size
        self.r_dim = self.r_n_rows + self.r_n_cols
        self.is_reduced = True

    def _equations(self, x):
        eqs(x, self.r_dseq_rows, self.r_dseq_cols, \
            self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.residuals)

    def _jacobian(self, x):
        jac(x, self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.r_n_cols, self.J_T)

    def _cost_gradient(self, x):
        self._equations(x)
        self._jacobian(x)
        return 0.5 * np.dot(self.residuals, self.residuals), \
                np.dot(self.J_T, self.residuals)

    def _residuals_jacobian(self, x):
        self._equations(x)
        self._jacobian(x)
        return self.residuals, self.J_T

    def _equations_rvalue(self, x):
        self._equations(x)
        return self.residuals

    def _jacobian_rvalue(self, x):
        self._jacobian(x)
        return self.J_T

    def _hessian_vector_product(self, x, p):
        return np.dot(self.J_T.T, np.dot(self.J_T, p))

    def _hessian(self, x):
        self._jacobian(x)
        return self.J_T @ self.J_T.T

    def _initialize_problem(self):
        if ~self.is_reduced:
            self.degree_degeneration()
        self.J_T = np.empty((self.r_dim, self.r_dim), dtype=np.float64)
        self.residuals = np.empty(self.r_dim, dtype=np.float64)

    def _set_solved_problem(self, solution):
        self.r_xy = solution.x
        self.final_result = solution.fun
        self.r_x = self.r_xy[:self.r_n_rows]
        self.r_y = self.r_xy[self.r_n_rows:]
        self.x = self.r_x[self.r_invert_dseq_rows]
        self.y = self.r_y[self.r_invert_dseq_cols]

    def _clean_problem(self):
        self.J_T = None
        self.residuals = None

    def solve_trust_regions(self, initial_guess=None, method='trust-ncg', \
                        display=False, maxiter=1000):
        self._initialize_problem()
        if initial_guess is None:
            self.r_x = self.r_dseq_rows * self.r_dseq_rows / self.n_edges
            self.r_y = self.r_dseq_cols * self.r_dseq_cols / self.n_edges
            x0 = np.concatenate((self.r_x, self.r_y)).astype(np.float64)
        elif initial_guess == 'random':
            x0 = np.random.rand(self.r_dim).astype(np.float64)
        else:
            x0 = initial_guess.astype(np.float64)
        opz = {'disp':display, 'maxiter':maxiter}
        if method == 'trust-ncg' or method == 'trust-krylov':
            res = opt.minimize(self._cost_gradient, x0, method=method, \
                               jac=True, hessp=self._hessian_vector_product, \
                               options=opz)
        elif method == 'trust-exact':
            res = opt.minimize(self._cost_gradient, x0, method=method, \
                               jac=True, hess=self._hessian, \
                               options=opz)
        self._set_solved_problem(res)
        del(res)
        self._clean_problem()

    def solve_root(self, initial_guess=None, method='hybr', scale=None):
        self._initialize_problem()
        if initial_guess is None:
            self.r_x = self.r_dseq_rows * self.r_dseq_rows / self.n_edges
            self.r_y = self.r_dseq_cols * self.r_dseq_cols / self.n_edges
            x0 = np.concatenate((self.r_x, self.r_y)).astype(np.float64)
        elif initial_guess == 'random':
            x0 = np.random.rand(self.r_dim).astype(np.float64)
        else:
            x0 = initial_guess.astype(np.float64)
        opz = {'col_deriv':True, 'diag':scale}
        res = opt.root(self._residuals_jacobian, x0, method=method, jac=True, \
                        options=opz)
        self._set_solved_problem(res)
        del(res)
        self._clean_problem()

    def solve_least_squares(self, initial_guess=None, method='trf', \
                            scale=1.0, tr_solver='lsmr', disp=False):
        self._initialize_problem()
        if initial_guess is None:
            self.r_x = self.r_dseq_rows * self.r_dseq_rows / self.n_edges
            self.r_y = self.r_dseq_cols * self.r_dseq_cols / self.n_edges
            x0 = np.concatenate((self.r_x, self.r_y)).astype(np.float64)
        elif initial_guess == 'random':
            x0 = np.random.rand(self.r_dim).astype(np.float64)
        else:
            x0 = initial_guess.astype(np.float64)
        res = opt.least_squares(self._equations_rvalue, x0, method=method, \
                                jac=self._jacobian_rvalue, x_scale=scale, \
                                tr_solver=tr_solver, verbose=disp)
        self._set_solved_problem(res)
        del(res)
        self._clean_problem()

    def solve_partial(self, constrained_layer=0):
        if ~self.is_reduced:
            self.degree_degeneration()
        if constrained_layer == 0:
            self.r_xy = np.concatenate((self.r_dseq_rows / self.n_cols, \
                                        np.ones(self.r_n_cols)))
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
            self.x = self.r_x[self.r_invert_dseq_rows]
            self.y = self.r_y[self.r_invert_dseq_cols]
        else:
            self.r_xy = np.concatenate((np.ones(self.r_n_rows), \
                                        self.r_dseq_cols / self.n_rows))
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
            self.x = self.r_x[self.r_invert_dseq_rows]
            self.y = self.r_y[self.r_invert_dseq_cols]
            
    def generate_p_adjacency(self):
        self.p_adjacency = probability_matrix(self.x, self.y)


class biWcm_leastSquares(bicm_leastSquares):
    def __init__(self, strength_sequence, n_rows, n_cols):
        super().__init__(strength_sequence, n_rows, n_cols)

    def _equations(self, x):
        eqs_W(x, self.r_dseq_rows, self.r_dseq_cols, \
            self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.residuals)

    def _jacobian(self, x):
        jac_W(x, self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.r_n_cols, self.J_T)


class reformed_biWcm_leastSquares(bicm_leastSquares):
    def __init__(self, strength_sequence, n_rows, n_cols):
        super().__init__(strength_sequence, n_rows, n_cols)

    def _equations(self, x):
        reform_eqs_W(x, self.r_dseq_rows, self.r_dseq_cols, \
            self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.residuals)

    def _jacobian(self, x):
        reform_jac_W(x, self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.r_n_cols, self.J_T)




