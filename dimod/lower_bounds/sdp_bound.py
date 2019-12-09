# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# =============================================================================
"""
Lower bounds on the energy of an Binary Quadratic Model using semidefinite programs.

"""


import numpy as np
import dimod
import scipy.linalg
import scipy.sparse
import time


def f_r(Q, V):
    # Original objective function of the semidefinite program: equation (7) of [#gvppr]_.

    sqlen = np.sum(V ** 2, axis=1)
    invlength = 1./np.sqrt(sqlen)
    Vn = np.dot(np.diag(invlength), V)
    M = np.matmul(Q, Vn)
    return np.sum(Vn * M), sqlen


def f_e(Q, V, eps, delta):
    # Objective function of the semidefinite program modified to include a normalization penalty:
    # equation (14) of [#gvppr]_.
    fr, sqlen = f_r(Q, V)
    penalty = (sqlen - 1) ** 2
    d = delta ** 2 - np.maximum(0, 1 - sqlen) ** 2  # Eq. (13)
    return fr + np.sum(penalty / d) / eps


def gradients_f_r(Q, V):
    # Gradient of the original objective function of the SDP: page 9 of [#gvppr]_.

    # normalize:
    sqlen = np.sum(V ** 2, axis=1)
    invlength = 1./np.sqrt(sqlen)
    Vn = np.dot(np.diag(invlength), V)

    # compute:
    M = np.matmul(Q, Vn)
    s = np.sum(Vn * M, axis=1)
    gradients = np.matmul(Q - np.diag(s), Vn)
    gradients = 2*np.matmul(np.diag(1/np.sqrt(sqlen)), gradients)
    return gradients


def gradients_f_e(Q, V, eps, delta):
    # Gradient of the original objective function of the SDP: page 8 of [#gvppr]_.

    gradients_fr = gradients_f_r(Q, V)
    sqlen = np.sum(V ** 2, axis=1)
    d = delta ** 2 - np.maximum(0, 1 - sqlen) ** 2  # Eq. (13)
    sqlenm1 = sqlen - 1  # sq. len minus 1
    A = sqlenm1 / d
    B = 1 - sqlenm1 * np.maximum(-sqlenm1, 0) / d
    gradients = gradients_fr + np.matmul(np.diag(4 * A * B / eps), V)
    return gradients


def lagrange_multipliers(Q, V):
    # Exact formula for the lagrange mutlipliers of the SDP, obtained from KKT conditions.
    # Equation (10) [#gvppr]_.

    M = np.matmul(Q, V)
    mults = -np.sum(V * M, axis=1)
    return mults


def sdp_lower_bound_from_V(Q, V):
    # Lower bound for the Ising problem, valid for any feasible V (i.e. even if V is not optimal).
    # page 13 of [#gvppr]_.
    n = Q.shape[0]
    la = lagrange_multipliers(Q, V)
    M = Q + np.diag(la)
    eigs = scipy.linalg.eigh(M)[0]
    min_eig = eigs[0]

    tr = np.sum(V * np.matmul(Q, V))
    lower_bound = tr + n * np.minimum(0, min_eig)
    return lower_bound, min_eig


def gradient_descent(Q, r, eps, delta):
    """
    Solve the low-rank SDP problem

        min tr(QVV^T) s.t. {(VV^T)_ii = 1, rank(V) = r}

    using gradient descent.
    """

    # Choose a random starting point:
    n = Q.shape[0]
    V = np.random.uniform(low=-1, high=1, size=(n, r))
    # normalize random starting point to have norms ~ 1
    sqlen = np.sum(V ** 2, axis=1)
    V = np.dot(np.diag(1. / np.sqrt(sqlen)), V)

    finished = False
    best_obj = f_e(Q, V, eps, delta)
    step = 0
    alpha = 1.
    max_sqlen = np.max(np.sum(V ** 2, axis=1))
    print(step, best_obj, alpha, 0, max_sqlen)
    converged = False
    while not finished:
        step += 1
        g = gradients_f_e(Q, V, eps, delta)
        newV = V - alpha * g
        new_obj = f_e(Q, newV, eps, delta)
        while new_obj >= best_obj:
            alpha /= 2
            newV = V - alpha * g
            new_obj = f_e(Q, newV, eps, delta)
            print('\t', new_obj, alpha)
            if alpha < 1e-10:
                return V, converged

        V = newV
        best_obj = new_obj
        max_grad = np.max(np.abs(g))
        converged = max_grad < 1e-5
        finished = converged
        max_sqlen = np.max(np.sum(V ** 2, axis=1))
        if step % 100 == 0:
            print(step, best_obj, alpha, max_grad, max_sqlen)

    return V, converged


def barzilai_borwein(Q, r, eps, delta, grad_tol, bound_tol, max_iters=None, max_time=None, verbose=True):
    """
    Solve the low-rank SDP problem

        min tr(QVV^T) s.t. {(VV^T)_ii = 1, rank(V) = r}

    using the Barzilai-Borwein variant of gradient descent.

    Following the notation in [#gvppr]_, V is a matrix with columns = {v_1, v_2, ..., v_n}, where v_i is a vector in
    R^r.
    """

    # Parameters of the algorithm
    n = Q.shape[0]                                                          # problem dimension
    alpha = 1/n                                                             # initial step size
    int_problem = np.max(2*Q % 1) == 0 and np.max(np.diag(Q) % 1) == 0      # problem coefficients are integers

    # Choose a random starting point:
    V = np.random.uniform(low=-1, high=1, size=(n, r))
    # normalize random starting point to have norms ~ 1
    sqlen = np.sum(V ** 2, axis=1)
    V = np.dot(np.diag(1. / np.sqrt(sqlen)), V)
    best_obj = f_e(Q, V, eps, delta)

    if verbose:
        max_sqlen = np.max(np.sum(V ** 2, axis=1))
        print("(step, objective, alpha, norm_grad, max_sqlen, lower_bound)")
        print(0, best_obj, alpha, 0, max_sqlen)

    # Start the algorithm.
    step = 0
    finished = False
    start_time = time.time()
    converged = False
    while not finished:
        step += 1
        g = gradients_f_e(Q, V, eps, delta)
        if step == 1:
            # do a line search on first step to find alpha
            # TODO: look at better line search implementations for first step (moves far from sphere)
            newV = V - alpha * g
            new_obj = f_e(Q, newV, eps, delta)
            if new_obj >= best_obj:
                # decrease alpha.
                while new_obj >= best_obj:
                    alpha /= 2
                    newV = V - alpha * g
                    new_obj = f_e(Q, newV, eps, delta)
            else:
                # increase alpha
                while new_obj < best_obj:
                    best_obj = new_obj
                    alpha *= 2
                    newV = V - alpha * g
                    new_obj = f_e(Q, newV, eps, delta)
                alpha /= 2
            alpha /= 2
        else:
            # after first step, do B-B updates to find alpha
            g_diff = g - g_old
            V_diff = V - V_old
            alpha = np.sum(V_diff*g_diff)/np.sum(g_diff**2)

        # update:
        V_old = V
        V = V - alpha*g
        g_old = g

        norm_grad = np.sqrt(np.sum(g**2))
        converged = norm_grad < grad_tol
        finished = converged
        if max_iters and step == max_iters:
            finished = True
        if max_time and time.time()-start_time > max_time:
            finished = True
        if step % 100 == 0:
            # compute an explicit lower bound
            lower_bound = sdp_lower_bound_from_V(Q, V)[0]
            if int_problem and np.ceil(lower_bound) == np.ceil(best_obj):
                finished = True
            elif lower_bound > best_obj - bound_tol:
                finished = True

        # debugging:
        if verbose and (finished or step % 100 == 0):
            max_sqlen = np.max(np.sum(V ** 2, axis=1))
            best_obj = f_e(Q, V, eps, delta)
            print(step, best_obj, alpha, norm_grad, max_sqlen, sdp_lower_bound_from_V(Q, V)[0])

    return V, converged


def augmented_ising(bqm_orig):
    """
    Given a binary quadratic model, convert it to an Ising model with no linear terms.

    To remove linear terms, augment the Ising model with an auxiliary variable x_0 that is assumed to take the value
    z_0 = 1. Then for any other variable z_i with linear coefficient h_i in the original Ising model, define a
    quadratic coefficient J_{0,i} = h_i in the new Ising model.

    The result is returned as a symmetric numpy array (matrix) of quadratic terms.
    """

    if bqm_orig.vartype == dimod.BINARY:
        bqm = bqm_orig.change_vartype(dimod.SPIN)
    else:
        bqm = bqm_orig

    n = len(bqm.variables)
    h, (row, col, data), offset = bqm.to_numpy_vectors(variable_order=bqm.variables)
    J = scipy.sparse.coo_matrix((data, (row, col)), shape=[n, n]).toarray()

    # remove linear terms:
    if np.any(h):
        J = np.concatenate([np.zeros((1, n + 1)), np.concatenate([h.reshape(-1, 1), J], axis=1)], axis=0)
        n += 1

    # symmetrize J
    J = (J + np.transpose(J)) / 2
    return J, offset


def sdp_lower_bound(bqm, max_time=None, max_iters=10000, verbose=False):
    """
    Find a lower bound on the ground state bound of a binary quadratic model using a semidefinite programming
    relaxation [#gw]_.


    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            A binary polynomial.

        max_time: (integer, default=None):
            Maximum number of seconds allowed for the whole algorithm.

        max_iters: (integer, default=10000):
            Maximum number of gradient descent steps for each semidefinite matrix rank. Choosing a smaller number of
            steps may speed up the algorithm by terminating smaller rank problems earlier, but with too few steps the
            algorithm may fail to converge.

        verbose (boolean, default = False):
            If verbose=True, print details of the algorithm.


    Returns:
        float: lower bound on the energy of the bqm.

        dict: additional information resulting from the lower bound computation, including the following keys:
            'state': values of the variables in the LP relaxation.
            'optimality': whether or not the program was solved to optimality.


    This method solves a semidefinite programming relaxation of an Ising problem. More precisely, an Ising model of the
    form

    (1) \min_\{z_i \in {-,1}\} \sum_{i < j} z_i J_ij z_j

    is relaxed to a semidefinite program of the form

    (2) \min_\{Z p.s.d.} tr(JZ) subject to Z_ii = 1.

    If Z has rank 1, then Z = zz^T for some vector z \in {1,-1}^n, and since tr(JZ) = tr(Jzz^T) = z^TJz,
    the optimization (2) is equivalent to the original Ising model problem (1). This lower bound on (1) obtained by
    solving (2) is known as the Goemans-Williamson bound [#gw]_.

    To optimize (2), we follow the method in [#gvppr]_. In that method Z is assumed to have rank r,
    for iteratively increasing r, until an optimal solution is found. The optimization at rank r is acheived by
    applying gradient descent to an objective function that is modified to include penalties for violating the
    constraints Z_ii = 1. A particularly fast variation of gradient descent known as the Barzilai-Borwein method
    [#bb]_ is used.

    Requires the package scipy.

    .. [#gvppr] Grippo, L., Palagi, L., Piacentini, M., Veronica Piccialli, V., and Rinaldi, G.: "SpeeDP: an
    algorithm to compute SDP bounds for very large Max-Cut instances". Math. Program. 136(2), 353-373 (2012).

    .. [#gw] Goemans, M.X. and Williamson, D.P.: "Improved approximation algorithms for maximum cut and satisfiability
    problems using semidefinite programming". J. ACM 42(6), 1115–1145 (1995).

    .. [#bb] Barzilai, J. and  Borwein, J.M.: "Two point step size gradient method". IMA J. Numer. Anal. 8,
    141–148 (1988).

    """

    if not bqm:
        bound = bqm.offset
        info = {'state': None}
        return bound, info

    Q, offset = augmented_ising(bqm)

    # Set algorithm parameters
    n = Q.shape[0]
    r_min = max(min(n, 3), int(np.floor(n ** (1 / 3))))         # minimum rank of SDP relaxation
    r_max = int(np.floor((np.sqrt(1 + 8 * n) - 1) / 2))         # maximum rank of SDP relaxation
    delta = 0.25                                                # vector normalization slack
    eps = 1e3 / delta                                           # vector normalization penalty weight (inverse)
    # epsilon comes with a speed/bound tightness trade-off. Large eps is faster and looser.
    eig_tol = 1e-3                                              # nonnegative eigenvalue tolerance

    kwargs = {'delta': delta, 'eps': eps, 'grad_tol': 1e-3, 'bound_tol': 1e-3, 'max_iters': max_iters,
              'verbose': verbose}

    # Run the incremental rank algorithm.
    finished = False
    lower_bound, V = -np.Inf, []
    best_bound = -np.Inf
    min_eig = []
    r = r_min
    start_time = time.time()
    converged = False
    while not finished:
        if max_time:
            kwargs['max_time'] = max_time + start_time - time.time()

        # run gradient descent.
        V, converged = barzilai_borwein(Q, r, **kwargs)

        lower_bound, min_eig = sdp_lower_bound_from_V(Q, V)
        best_bound = max(best_bound, lower_bound)
        if verbose:
            print("r:", r, " min_eig:", min_eig, " bound: ", lower_bound)

        if (converged and min_eig >= -eig_tol) or r >= r_max:
            finished = True
        if max_time and time.time() - start_time > max_time:
            if verbose:
                print("Time limit exceeded.")
            finished = True

        # increase rank.
        r = min(int(np.floor(1.5 * r)), r_max)

    optimality = converged and min_eig >= -eig_tol
    if optimality:
        # V is valid solution to orig problem
        # (if V is feasible, then the bound f_r is at least as good as sdp_lower_bound_from_V)
        lower_bound = f_r(Q, V)[0]
        best_bound = max(best_bound, lower_bound)

    info = {'state': V, 'optimality': optimality}
    if verbose:
        print("Optimality: {}".format(optimality))
        print("Offset: {}, final bound: {}".format(offset, best_bound+offset))
    return lower_bound + offset, info
