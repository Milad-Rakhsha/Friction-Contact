# =============================================================================
# SIMULATION-BASED ENGINEERING LAB (SBEL) - http://sbel.wisc.edu
# University of Wisconsin-Madison
#
# Copyright (c) 2020 SBEL
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# at https://opensource.org/licenses/BSD-3-Clause
#
# =============================================================================
# Contributors: Nic Olsen, Milad Rakhsha
# =============================================================================
"""
The integration of the non-smooth mechanical system using complementarity approach.
The integration loop involve decoupling of the normal force and Tangential force calculation
to obtain a compatible set of solution.
"""

import sys, time
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from buildB import buildB
from solvers import solver


def error(App, Exa):
    return norm(App - Exa) / norm(Exa) * 100


def regularization_matrix(n_contacts, normal=True, friction=False):
    weights = np.zeros(3 * n_contacts)
    if normal:
        weights[0::3] = 1
    if friction:
        weights[1::3] = 1
        weights[2::3] = 1

    W = np.eye(3 * n_contacts, dtype='d')
    W = np.multiply(W, weights)
    print("|W|=", np.sum(W))
    return np.matmul(W.T, W)


def QOCC_Augmented_Lagrangian(Cost_N, Cost_p, Constraint_N, Constraint_p, x0, QP_solver, contact_static_fric, tolerance,
                              eta0=1e-4, eta_max=1e5, tau_eta=10):
    '''
    A helper function to solve a Quadratic Optimization q=1/2 x^T Cost_N x + Cost_p^T x
    with Conic Constraints (QOCC) and
    additional equality constraints in forms of A(x-x0)=0 and b^T(x-x0)=0
    It can be shown that the equality constraints of this sort can be incorporated into the quadratic cost function
    with coefficient of Constraint_N= 2*(A^T A+ b b^T), Constraint_p= -2*x0^T *(A^T A+ b b^T)
    Note : the user has to provide the Constraint_N, and Constraint_p according to the equality constraints
    of the problem
    '''

    x_star = x0.copy()
    x_new = x0.copy()
    eta = eta0
    iter_1 = 0
    while eta < eta_max:
        N_mat = 2 * Cost_N + 2 * Constraint_N * eta
        p_vec = Cost_p + eta * Constraint_p
        x_old = x_new.copy()
        x_new, totalnumIter = QP_solver.solve(N_mat, p_vec, x_old, contact_static_fric, tolerance)
        Cons_val = np.matmul((x_new - x_star).T, np.matmul(Constraint_N, (x_new - x_star)))

        iter_1 += 1
        eta *= tau_eta
        if iter_1 == 1:
            print("Constraint=%3e, eta=%3e, f_norm=%3e, numIter=%d, inner_iter=%d" % \
                  (Cons_val, eta, norm(x_new), iter_1, totalnumIter))
        if Cons_val < 1e-12 and iter_1 > 5:
            break
    # 1/2*np.matmul((x_new).T, np.matmul(Cost_N, (x_new)))+np.matmul((x_new).T,Cost_p)

    print("Constraint=%3e, eta=%3e, f_norm=%3e, numIter=%d, inner_iter=%d" % \
          (Cons_val, eta, norm(x_new), iter_1, totalnumIter))
    print("x_star=\n", x_star.reshape(Cost_p.shape[0] // 3, 3))
    print("x_new=\n", x_new.reshape(Cost_p.shape[0] // 3, 3))
    print("sum=\n", np.sum(x_new.reshape(Cost_p.shape[0] // 3, 3), 0))
    return x_new


def integrate(q, v, params, random_initial=False, warm_x=np.array([])):
    buildB_time = form_time = solve_time = update_time = 0

    t1 = time.time()
    B, phi, n_contacts, c_pos, contact_static_fric, contact_pairs = buildB(q, params)
    t2 = time.time()
    buildB_time = t2 - t1

    QP_solver = solver(params.solver)
    # Mask out fixed bodies from all equations of motion
    F_ext = np.multiply(params.F_ext, [1 if not params.fixed[int(i / 6)] else 0 for i in range(6 * params.nb)])
    totalnumIter = 0
    if n_contacts != 0:
        t1 = time.time()
        A1 = np.matmul(np.matmul(B, params.M_inv), B.T)
        if params.unique:
            p = 1.0 / (params.dt ** 2) * np.kron(phi, np.array([1, 0, 0])) + \
                np.dot(B, 1.0 / params.dt * v + np.dot(params.M_inv, params.F_ext))
        else:
            N = A1
            p = 1.0 / (params.dt ** 2) * np.kron(phi, np.array([1, 0, 0])) + \
                np.dot(B, (1.0 / params.dt) * v + np.dot(params.M_inv, F_ext))

        if random_initial:
            f = 100 * np.random.ranf(3 * n_contacts)
        else:
            # f = warm_x.copy() if warm_x.shape[0] == n_contacts else np.zeros(3*n_contacts, dtype='d')
            f = np.zeros(3 * n_contacts, dtype='d')
            f.shape = 3 * n_contacts

        t2 = time.time()
        form_time = t2 - t1

        t1 = time.time()
        if params.unique:
            eps = params.eps_0
            mu = params.mu_tilde
            WW = regularization_matrix(n_contacts, normal=True, friction=False)
            # print (WW)
            iter = 0

            while mu > params.mu_star and iter < 1:
                A3 = mu * WW
                N = A1 + A3
                val = np.linalg.eigvals(N)
                f, numIter = QP_solver.solve(N, p, f, contact_static_fric, eps)
                # Update penalty term and tolerance
                eps = max(params.tau_eps * eps, params.solver["tolerance"])
                mu = params.tau_mu * mu
                totalnumIter = totalnumIter + numIter
                iter = iter + 1

            # print("%s, tikhonov iter=%d, alpha=%e, eps=%e, total_solver_iter=%d"%\
            #      (params.solver,iter,mu,eps,totalnumIter))
            print("alpha=%3e, eps=%3e, numIter=%d" % (mu, eps, totalnumIter))
            # print("tikhonov: {}".format(iter))
        else:
            f, totalnumIter = QP_solver.solve(N, p, f, contact_static_fric, params.solver["tolerance"])

        Cost = QP_solver.f_eval(f, N, p)
        print("Original cost function:%3e" % Cost)
        mu = params.mu_tilde

        print("sum=\n", np.sum(f.reshape(f.shape[0] // 3, 3), 0))
        pv = p.reshape(3 * n_contacts, 1)
        X = np.matmul(N.T, N) + np.matmul(pv, pv.T)
        # print(p.shape,pv.shape)
        # print(np.matmul(pv,pv.T).shape,X.shape)
        f_star = f.copy()
        f_new = f.copy()
        WW_n = regularization_matrix(n_contacts, normal=True, friction=False)
        f_N = QOCC_Augmented_Lagrangian(Cost_N=WW_n,
                                        Cost_p=np.zeros(3 * n_contacts),
                                        Constraint_N=X,
                                        Constraint_p=-2 * np.matmul(f_star.T, X),
                                        x0=f_star,
                                        QP_solver=QP_solver,
                                        contact_static_fric=contact_static_fric,
                                        tolerance=params.solver["tolerance"],
                                        eta0=params.eta0, eta_max=params.eta_max, tau_eta=params.tau_eta)
        Cost_original_N = QP_solver.f_eval(f_N, N, p)
        Cost_N = QP_solver.f_eval(f_N, 2 * WW_n, np.zeros(3 * n_contacts))
        print("Original cost function with f_N error: %3e\nNormal compatible cost function: %3e"
              % (error(Cost_original_N, Cost), Cost_N))
        WW_T = regularization_matrix(n_contacts, normal=False, friction=True)
        f_T = QOCC_Augmented_Lagrangian(Cost_N=WW_T,
                                        Cost_p=np.zeros(3 * n_contacts),
                                        Constraint_N=X
                                                     + np.matmul(WW_n.T, WW_n),
                                        Constraint_p=
                                        -2 * np.matmul(f_star.T, X)
                                        - 2 * np.matmul(f_N.T, np.matmul(WW_n.T, WW_n)),
                                        x0=f_N,
                                        QP_solver=QP_solver,
                                        contact_static_fric=contact_static_fric,
                                        tolerance=params.solver["tolerance"],
                                        eta0=params.eta0, eta_max=params.eta_max, tau_eta=params.tau_eta)

        Cost_original_T = QP_solver.f_eval(f_T, N, p)
        Cost_N_ft = QP_solver.f_eval(f_T, 2 * WW_n, np.zeros(3 * n_contacts))
        Cost_T = QP_solver.f_eval(f_T, 2 * WW_T, np.zeros(3 * n_contacts))
        print(
            "Original cost function with f_T error: %3e\nNormal compatible cost function with f_T error: %3e\nTangential Cost function: %3e"
            % (error(Cost_original_T, Cost), error(Cost_N_ft, Cost_N), Cost_T))

        f = f_T
        # print("%s, total_solver_iter=%d"% (params.solver["type"], numIter))
        t2 = time.time()
        solve_time = t2 - t1

        F_contact = np.dot(B.T, f)

        F = np.add(F_contact, F_ext)
    else:
        f = np.zeros((0, 0))
        F = F_ext

    new_a = np.dot(params.M_inv, F)
    new_v = np.add(v, params.dt * new_a)
    new_q_dot = np.zeros(7 * params.nb)

    t1 = time.time()
    for i in range(params.nb):
        # Write linear velocities as is
        new_q_dot[i * 7: i * 7 + 3] = new_v[i * 6: i * 6 + 3]

        # Apply quaternion identity: q_dot = 0.5 * omega * q for each omega
        rot = q[i * 7 + 3: i * 7 + 7]
        w = new_v[i * 6 + 3: i * 6 + 6]
        rot_dot = np.zeros(4)
        rot_dot[0] = -0.5 * (rot[1] * w[0] + rot[2] * w[1] + rot[3] * w[2])
        rot_dot[1] = 0.5 * (rot[0] * w[0] - rot[2] * w[2] + rot[3] * w[1])
        rot_dot[2] = 0.5 * (rot[0] * w[1] + rot[1] * w[2] - rot[3] * w[0])
        rot_dot[3] = 0.5 * (rot[0] * w[2] - rot[1] * w[1] + rot[2] * w[0])
        new_q_dot[i * 7 + 3: i * 7 + 7] = rot_dot

    new_q = q + params.dt * new_q_dot

    # NOTE for safety, normalize quaternions every step
    for i in range(params.nb):
        rot = q[i * 7 + 3: i * 7 + 7]
        if abs(norm(rot, 2) - 1) > 0.01:
            q[i * 7 + 3: i * 7 + 7] = rot / norm(rot, 2)
    f.shape = (n_contacts, 3)

    t2 = time.time()
    update_time = t2 - t1

    print(
        "Build = %.2e, Form = %.2e, Solve = %.2e, Update = %.2e, " % (buildB_time, form_time, solve_time, update_time))
    return new_q, new_v, new_a, c_pos, f, B, contact_pairs, phi, totalnumIter
