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
The cone complementarity solvers for contact dynamics problems.
"""

import numpy as np

class solver():
    def __init__(self,solver_params,const_N=np.array([])):
        self.const_Normals=const_N
        self.params=solver_params
        self.type=solver_params["type"]
        self.tolerance=solver_params["tolerance"]
        self.max_iterations=solver_params["max_iterations"]
        if(self.const_Normals.shape[0]>0):
            print("Using constant normals of ",const_N )


    def solve(self,N,p,x0,friction,eps):
        if (self.type=="Gauss_Seidel"):
            return self.Gauss_Seidel(N, p, friction, eps, self.max_iterations, x0, self.params["GS_omega"], self.params["GS_lambda"])
        elif (self.type=="Jacobi"):
            return self.Jacobi(N, p, friction, eps, self.max_iterations, x0, self.params["Jacobi_omega"], self.params["Jacobi_lambda"])
        elif (self.type=="APGD"):
            return self.apgd(N, p, friction, eps, self.max_iterations, x0)
        else:
            return self.apgd_ref(N, p, friction, eps, self.max_iterations, x0)


    def Jacobi(self,N, p, friction, tolerance, max_iterations, x, omega_=0.3, lambda_=0.9, g=1e-6):
        n = x.shape[0]
        Nc = int(x.shape[0] / 3)
        B = np.zeros(Nc*3)
        for i in range(Nc):
            s_i=3*i
            e_i=3*i+3
            g_i=np.sum( np.diag( N[s_i:e_i, s_i:e_i]) ) / 3.0
            if(g_i!=0):
                B[s_i:e_i] = np.ones(3) / g_i
            else:
                B[s_i:e_i] = np.ones(3)


        for iter in range(max_iterations):
            x_old = x.copy()
            # if(iter%1==0):
            #     print("norm x:", np.linalg.norm(x))
            grad = self.f_grad(x, N, p)
            x_hat=self.project(x - omega_ * B * grad , friction)
            x=lambda_*x_hat+(1.0-lambda_)*x
            res=np.linalg.norm(x_old-x)
            # res = self.r4_error(x, N, p, friction, g)
            # Sufficient accuracy reached
            if res < tolerance:
                break

        return x, iter

    def Gauss_Seidel(self,N, p, friction, tolerance, max_iterations, x, omega_=0.9, lambda_=0.9, g=1e-6):
        n = x.shape[0]
        Nc = int(x.shape[0] / 3)

        for iter in range(max_iterations):
            x_old=x.copy()
            for i in range(Nc):
                s_i=3*i
                e_i=3*i+3
                g_i=np.sum( np.diag( N[s_i:e_i, s_i:e_i ]) ) / 3.0
                # Bi=np.ones(3)/g_i
                if(g_i!=0):
                    Bi = np.ones(3) / g_i
                else:
                    Bi = np.ones(3)

                grad = self.f_grad(x, N, p)
                x_hat_i=self.project(x[s_i:e_i] -omega_ *  Bi * grad[s_i:e_i] ,friction)
                x[s_i:e_i]=lambda_*x_hat_i+(1.0-lambda_)*x[s_i:e_i]
            res=np.linalg.norm(x_old-x)
            # res = r4_error(x, N, p, friction, g)
            # Sufficient accuracy reached
            if res < tolerance:
                # print("Gauss_Seidel converged to {} after {} iters.".format(res, iter))
                break

        return x, iter


    # Assumes N is symmetric
    def apgd_ref(self,N, p, friction, tolerance, max_iterations, x0, g=1e-6):
        n=x0.shape[0]
        g=1.0/np.power(n,2)
        residual = 10e30
        # gamma=np.zeros(n, dtype='d')
        gamma=x0.copy()
        # (2) gamma_hat_0 = ones(nc,1)
        gamma_hat=np.ones(n, dtype='d')
        # (3) y_0 = gamma_0
        y = gamma
        # (4) theta_0 = 1
        theta,thetaNew = 1.0,1.0
        Beta = 0.0
        obj1,obj2=0.0,0.0

        # (5) L_k = norm(N * (gamma_0 - gamma_hat_0)) / norm(gamma_0 - gamma_hat_0)
        tmp = gamma - gamma_hat
        L = np.linalg.norm(tmp)
        if (L > 0):
            tmp=np.dot(N, tmp)
            L = np.linalg.norm(tmp) / L
        else:
            L = 1.0
        # (6) t_k = 1 / L_k
        if (L > 0):
            t = 1.0 / L
        else:
            L = 1
            t = 1
        #(7) for k := 0 to N_max
        for iter in range(max_iterations):
            # (8) g = N * y_k - r
            grad=self.f_grad(y, N, p)
            # (9) gamma_(k+1) = ProjectionOperator(y_k - t_k * g)
            gammaNew = self.project(y - t * grad, friction)
            # (10) while 0.5 * gamma_(k+1)' * N * gamma_(k+1) - gamma_(k+1)' * r
            # >= 0.5 * y_k' * N * y_k - y_k' * r + g' * (gamma_(k+1) - y_k) + 0.5 * L_k * norm(gamma_(k+1) - y_k)^2
            obj1 = self.f_eval(gammaNew, N, p)
            obj2 = self.f_eval(y, N, p)
            tmp = gammaNew - y
            obj2 = obj2 + np.dot(grad, tmp) + 0.5 * L * np.dot(tmp, tmp)
            while (obj1 >= obj2):
                        # (11) L_k = 2 * L_k
                        L = 2.0 * L
                        # (12) t_k = 1 / L_k
                        t = 1.0 / L
                        # (13) gamma_(k+1) = ProjectionOperator(y_k - t_k * g)
                        gammaNew = self.project(y - t * grad, friction)
                        # Update the components of the while condition
                        obj1 = self.f_eval(gammaNew, N, p)
                        obj2 = self.f_eval(y, N, p)
                        tmp = gammaNew - y  # Here tmp is equal to gammaNew - y
                        obj2 = obj2 + np.dot(grad, tmp) + 0.5 * L * np.dot(tmp, tmp)
                        # (14) endwhile
            # (15) theta_(k+1) = (-theta_k^2 + theta_k * Sqrt(theta_k^2 + 4)) / 2
            thetaNew = (-np.power(theta, 2.0) + theta * np.sqrt(np.power(theta, 2.0) + 4.0)) / 2.0
            # (16) Beta_(k+1) = theta_k * (1 - theta_k) / (theta_k^2 + theta_(k+1))
            Beta = theta * (1.0 - theta) / (np.power(theta, 2) + thetaNew)
            # (17) y_(k+1) = gamma_(k+1) + Beta_(k+1) * (gamma_(k+1) - gamma_k)
            yNew = gammaNew + Beta * (gammaNew - gamma)
            # (18) r = r(gamma_(k+1))
            res = self.r4_error(gammaNew, N, p, friction, g)
            # res=np.linalg.norm(gammaNew-gamma)

            # (19) if r < epsilon_min
            if (res < residual):
                # (20) r_min = r
                residual = res
                # (21) gamma_hat = gamma_(k+1)
                gamma_hat = gammaNew
                # (22) endif
            Nl=self.f_grad(gammaNew, N, p)
            objective_value = self.f_eval(Nl, N, p)
            if (residual <tolerance):
                #(24) break
                break
                #(25) endif
            # (26) if g' * (gamma_(k+1) - gamma_k) > 0
            if (np.dot(grad, gammaNew - gamma) > 0):
                # (27) y_(k+1) = gamma_(k+1)
                yNew = gammaNew
                # (28) theta_(k+1) = 1
                thetaNew = 1.0
                # (29) endif
            # (30) L_k = 0.9 * L_k
            L = 0.9 * L
            # (31) t_k = 1 / L_k
            t = 1.0 / L
            # Update iterates
            theta = thetaNew
            gamma = gammaNew
            y = yNew
            #(32) endfor

        #(33) return Value at time step t_(l+1), gamma_(l+1) := gamma_hat
        gamma = gamma_hat
        return gamma, iter

    # Assumes N is symmetric
    def apgd(self,N, p, friction, tolerance, max_iterations, x0, g=1e-6):
        n = x0.shape[0]

        theta_old = 1
        x_old = x0.copy()
        y = x0.copy()
        x_best = np.ones(n, dtype='d')
        r_min = np.inf

        # Initial estimate of Lipschitz L using arbitrary points 1 and 0
        L = np.linalg.norm(self.f_grad(np.ones((n,1)), N, p)) / n
        t = 1.0 / L
        for iter in range(max_iterations):
            grad = self.f_grad(y, N, p)
            x_new = self.project(y - t * grad, friction)

            # Update stepsize based on Lipschitz estimate L
            fy = self.f_eval(y, N, p)
            while self.f_eval(x_new, N, p) > fy + np.dot(grad, x_new - y) + 0.5 * L * np.linalg.norm(x_new - y)**2:
                L = 2 * L
                t = 1.0 / L
                x_new = self.project(y - t * grad, friction)

            theta_new = 0.5 * (-theta_old**2 + theta_old * np.sqrt(theta_old**2 + 4))
            beta = theta_old * (1 - theta_old) / (theta_old**2 + theta_new)
            y = x_new + beta * (x_new - x_old)

            r = self.r4_error(x_new, N, p, friction, g)

            # New best solution
            if r < r_min:
                r_min = r
                x_best = x_new

            # Sufficient accuracy reached
            if r < tolerance:
                # print("apgd: {}".format(iter))
                break

            # Adaptive restart
            if np.dot(grad, x_new - x_old) > 0:
                y = x_new.copy()
                theta_new = 1

            x_old = x_new
            theta_old = theta_new

            L = 0.9 * L
            t = 1.0 / L

        return x_best, iter

    def r4_error(self,x, N, p, friction, g):
        return np.linalg.norm( 1.0 / g * (x - self.project(x - g * self.f_grad(x, N, p), friction) ) )

    def f_eval(self,x, N, p):
        return 0.5 * np.dot(x, np.dot(N, x)) + np.dot(p, x)

    def f_grad(self,x, N, p):
        return np.dot(N, x) + p

    def project(self,x_in, friction):
        x = x_in.copy()

        if(self.const_Normals.shape[0]>0):
            for i in range(int(x.shape[0] / 3)):
                f = x[3*i : 3*i + 3]
                fn=self.const_Normals[i]
                ft = np.sqrt(f[1]**2 + f[2]**2)
                mu = friction[i]
                if ft <= mu *fn :
                    continue
                if mu != 0.0 and ft < -fn / mu:
                    f[0:3] = np.zeros(3)
                elif mu * fn < ft:
                    # f[0] = (fn + mu * ft) / (mu**2 + 1)
                    f[1] = f[1] * mu * fn / ft
                    f[2] = f[2] * mu * fn / ft

        else:
            for i in range(int(x.shape[0] / 3)):
                f = x[3*i : 3*i + 3]
                ft = np.sqrt(f[1]**2 + f[2]**2)
                mu = friction[i]
                if ft <= mu *f[0] :
                    continue
                if mu != 0.0 and ft < -f[0] / mu:
                    f[0:3] = np.zeros(3)
                elif mu * f[0] < ft:
                    f[0] = (f[0] + mu * ft) / (mu**2 + 1)
                    f[1] = f[1] * mu * f[0] / ft
                    f[2] = f[2] * mu * f[0] / ft

        return x
