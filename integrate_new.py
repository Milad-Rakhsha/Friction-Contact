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

import sys,time
from scipy import optimize
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from buildB import buildB
from solvers import solver
from OtherSolvers import find_slips

np.set_printoptions(precision=4,linewidth=150)
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format},threshold=sys.maxsize)
def error(App,Exa):
    return norm(App-Exa)/norm(Exa)*100


def swap(a,b):
    tmp=a
    a=b
    b=tmp
    return a,b

def rearrange_old_values(contact_pairs, params):
    old_contact_pairs=params.contact_pairs.copy()
    old_slips=params.slips.copy()
    old_normals=params.old_normals.copy()
    old_forces=params.old_forces.copy()
    new_nc=contact_pairs.shape[0]
    params.slips=np.zeros(new_nc)
    params.old_normals=np.zeros(new_nc)
    params.old_forces=np.zeros(new_nc*3)
    # print(contact_pairs)
    # print(old_contact_pairs)
    # import random
    # contact_pairs=np.random.shuffle(contact_pairs.copy())
    # np.random.shuffle(contact_pairs)
    # print(old_forces)
    # for c in range(contact_pairs.shape[0]):
    new_idx=0;
    retrieved=0
    for c in contact_pairs:
        # print (old_contact_pairs)
        # p=np.array_equal(old_contact_pairs[:,],c)
        # p=np.array(old_contact_pairs==contact_pairs[c,:],dtype=bool)
        p1=np.array(old_contact_pairs==c,dtype=bool)
        c[0],c[1]=swap(c[0],c[1])
        p2=np.array(old_contact_pairs==c,dtype=bool)
        c[0],c[1]=swap(c[0],c[1])
        # p= old_contact_pairs==c
        if(p1.shape and p2.shape):
             old_idx=np.where(p1[:,0] * p1[:,1] + p2[:,0] * p2[:,1])[0]
             # print ("contact ", c, "old_idx=",old_idx)
             # if there was a contact in the old system
             if(old_idx.shape[0]):
                 # params.old_forces[new_idx]=old_forces[old_idx[0]]
                 params.old_forces[3*new_idx:3*new_idx+3]=old_forces[3*old_idx[0]:3*old_idx[0]+3]
                 # print(old_forces[3*old_idx[0]:3*old_idx[0]+3])
                 params.old_normals[new_idx]=old_normals[old_idx[0]]
                 params.slips[new_idx]=old_slips[old_idx[0]]
                 retrieved+=1
             else:
                 print("new contact:", c)


        new_idx+=1
    if(new_idx):
        print("number of retrieved contact forces:%d, (%.2f)"%( retrieved, retrieved/new_idx*100) )
    if(params.compatibility=="old_slips"):
        print("Warm starting old slips- new rearranged slips=", norm(old_slips)-norm(params.slips))
    elif(params.compatibility=="old_normals"):
        # print("Warm starting old forces- new rearranged forces=", norm(old_forces)-norm(params.old_forces))
        print("Warm starting old ormals- new rearranged normals=", norm(old_normals)-norm(params.old_normals))



def regularization_matrix(n_contacts, normal=True, friction=False):
        weights= np.zeros(3*n_contacts)
        if(normal):
            weights[0::3]=1.0
        if(friction):
            weights[1::3]=1.0
            weights[2::3]=1.0

        W = np.eye(3*n_contacts, dtype='d')
        W=np.multiply(W,weights)
        print("|W|=", np.sum(W))
        return np.matmul(W.T, W)


def print_QOCC_Stats(Cons_val, Cost_original, Cost_Total, eta, iter, totalnumIter, res, inner_tol):
    print ("Cost_Constraint=%3e, Cost_original=%3e, eta=%3e, numIter=%d, totalnumIter=%d, inner_res=%.3g, inner_tol=%.3g"%\
    (Cons_val, Cost_original, eta, iter, totalnumIter, res, inner_tol))


def QOCC_Augmented_Lagrangian(Cost_N, Cost_p, Constraint_N, Constraint_p, A_E, x_s, x0, QP_solver,
                              contact_static_fric, tolerance,
                              eta0=1e-4,eta_max=1e5, tau_eta=10,verbose=True):
    '''
    A helper function to solve a Quadratic Optimization q=1/2 x^T Cost_N x + Cost_p^T x
    with Conic Constraints (QOCC) and
    additional equality constraints in forms of A(x-x_s)=0
    It can be shown that the equality constraints of this sort can be incorporated
    into the quadratic cost function
    with coefficient of Constraint_N= A^T*A, Constraint_p= -2*x0^T*A^T*A
    Note : the user has to provide the Constraint_N, and Constraint_p according to
    the equality constraints
    of the problem
    '''
    if(verbose):
        print("                    Augmented Lagrangian:")
    x_new=x0.copy()
    eta=eta0
    totalnumIter=0
    iter_1=0
    lam=0
    res=0
    QP_Max_iter=QP_solver.params["AL_QP_max_iterations"]
    while (eta<eta_max):
        # if(QP_solver.project_from_old_normal):
        #     QP_solver.project_from_old_normal=False
        # else:
        # QP_solver.project_from_old_normal=False

        N_mat=Cost_N+Constraint_N*eta
        p_vec=Cost_p+Constraint_p*eta

        Cost_total=QP_solver.f_eval(x_new-x_s,N_mat,p_vec)
        Cost_original=QP_solver.f_eval(x_new,Cost_N,Cost_p)
        Cons_val=np.max(A_E @ (x_new-x_s) )



        if(iter_1==0):
            QP_Max_iter=int(QP_Max_iter)
            tolerance=min(tolerance / (tau_eta), 1e-8)
            original_force_cost=Cost_original
        # else:
        #     QP_Max_iter=5000
        #     tolerance=1e-8

        if(verbose or iter_1==0):
            print_QOCC_Stats(Cons_val, Cost_original, Cost_total, eta,  iter_1, totalnumIter, res, tolerance)


        x_old=x_new.copy()

        x_new , this_iter, res= QP_solver.solve(N_mat, p_vec, x_old, contact_static_fric,
                                                    tolerance, QP_Max_iter )
        if(QP_solver.params["AL_use_AugmentedLagrangian"]):
            lam-=eta*A_E@(x_new-x_s)
            p_vec-= A_E.T@lam



        iter_1+=1
        totalnumIter+=this_iter
        eta*=tau_eta

        if (Cons_val<1e-10 and iter_1>10):
            break

    print_QOCC_Stats(Cons_val, Cost_original, Cost_total, eta, iter_1, totalnumIter, res, tolerance)

    if(original_force_cost<Cost_original):
        print("unsuccessfull compatibility solve. incompatible forces have smaller norm")
    #1/2*np.matmul((x_new).T, np.matmul(Cost_N, (x_new)))+np.matmul((x_new).T,Cost_p)
    # print ("Constraint=%3e, eta=%3e, f_norm=%3e, numIter=%d, inner_iter=%d"%\
    #     (Cons_val, eta, norm(x_new), iter_1, totalnumIter))
    # if(verbose):
        # print("x_star=\n", x_star.reshape(Cost_p.shape[0]//3,3))
        # print("x_new=\n", x_new.reshape(Cost_p.shape[0]//3,3))
        # # print("sum=\n", np.sum(x_new.reshape(Cost_p.shape[0]//3,3),0))
        # print("")
    return x_new

def integrate(q, v, params, random_initial=False, warm_x=np.array([])):
    buildB_time=form_time=solve_time=update_time=0
    t1=time.time()
    B, phi, n_contacts, c_pos, contact_static_fric, contact_pairs = buildB(q, params)
    t2=time.time()
    buildB_time=t2-t1

    # print("F before arrangment=", params.old_forces[::3])
    rearrange_old_values(contact_pairs, params)
    # print("F after arrangment=", params.old_forces[::3])
    # QP_solver = solver(params.solver, params.old_forces[::3],
    #                     project_from_old_normal= (params.compatibility=="old_normals" or params.compatibility=="incremental"))

    # we should be able to update without the need for old normal forces
    # but this is not working quit fine jut yet
    QP_solver = solver(params.solver, params.old_forces[::3], project_from_old_normal= True)
    # QP_solver = solver(params.solver)

    # Mask out fixed bodies from all equations of motion
    F_ext = np.multiply(params.F_ext, [1 if not params.fixed[int(i/6)] else 0 for i in range(6*params.nb)])
    totalnumIter=0
    if n_contacts != 0:
        print("# contacts= ", n_contacts)
        t1=time.time()
        A1 = np.matmul(np.matmul(B, params.M_inv), B.T)
        N = A1
        p = 1.0 / (params.dt**2) * np.kron(phi, np.array([1,0,0])) + \
            np.dot(B, (1.0 / params.dt) * v + np.dot(params.M_inv, F_ext))
        # print(N,p)
        if random_initial:
            f = 100* np.random.ranf(3*n_contacts)
        else:
            # f = params.old_forces.copy() # warm_x.copy() if warm_x.shape[0] == n_contacts else np.zeros(3*n_contacts, dtype='d')
            # f.shape = 3*n_contacts
            f=np.zeros(3*n_contacts, dtype='d')

        # print("1: size=", params.old_forces.shape, params.old_forces)
        t2=time.time()
        form_time=t2-t1

        t1=time.time()
        s_i=np.zeros(n_contacts)
        # print("f_contact=", norm(f[0::3]), norm(f[1::3]), norm(f[2::3]))
        f , totalnumIter, res= QP_solver.solve(N, p, f, contact_static_fric, params.solver["tolerance"])
        # print("f_contact=", norm(f[0::3]), norm(f[1::3]), norm(f[2::3]))
        incompatible_forces=f.copy()
        # print(f)
        # print("CD contact forces: \n", f.reshape(f.shape[0]//3,3))
        print("CD  iterations=%d,  residual= %.3g"%(totalnumIter,res))
        if(np.min(f[::3])<0):
            print("incorrect contact forces")
        # print("F CD=", f[::3])

        if(params.compatibility is not None):
            Cost=QP_solver.f_eval(f,N,p)
            # print("Original cost function:%3e"% Cost)
            f_star=f.copy()
            pv=p.reshape(3*n_contacts,1)
            Cost_N=regularization_matrix(n_contacts, normal=True,friction=True)
            ##### This is the old formulation where we used N to impose equality constraints
            # Constraint_N=2*np.matmul(N.T,N)# THIS IS REDUNDENT +np.matmul(pv,pv.T)
            ##### Ken suggested that using the B matrix should work too
            Constraint_N=2*np.matmul(B,B.T)#+0*np.matmul(pv,pv.T)# THIS IS REDUNDENT +np.matmul(pv,pv.T)
            Constraint_p=-np.matmul(Constraint_N,f_star)
            QP_solver = solver(params.solver)
            #############################################################################
            ### The goal of this section is to identify the set of KKT multipliers that
            ### represent slip. The idea is to for s_i=0 if T_i<mu* N_i else s_i=T_i/k_t
            Cost_p=np.zeros(3*n_contacts)
            x_guess=f_star # x in N(x-x_s)=0
            x_s=f_star     # x_s in N(x-x_s)=0
            if(warm_x.shape[0]>0):
                warm_x=warm_x.reshape(warm_x.shape[0]*warm_x.shape[1])
            print("Using",params.compatibility, "as the compatibility criteria" )

            if(params.compatibility=="old_normals"):
                ###Ken's first suggestion to deal with the mu*s terms
                QP_solver = solver(params.solver,params.old_forces[::3], project_from_old_normal=True)
                # W_N=regularization_matrix(n_contacts, normal=True,friction=False)
            elif(params.compatibility== "incremental"):
                # we should be able to update without the need for old normal forces
                # but this is not working quit fine jut yet,
                # use f_star for projecting to the new forces and params.old_forces
                QP_solver = solver(params.solver,
                                 const_N=None, project_from_old_normal=False, old_forces= params.old_forces ,#f_star #params.old_forces,
                                 project_from_old_normal_oldTangents=True)

                x_guess=np.ones(3*n_contacts)
                x_s=f_star-params.old_forces
                ## The increament of the tangent force satisfy the minimum norm
                # Cost_p[0::3]=params.old_forces[::3]
                # print(Constraint_N.shape,x_s.shape)
                Constraint_p=-np.matmul(Constraint_N,x_s)


                # print("f_star-params.old_forces=",x_s)
                # print("Constraint_p=",Constraint_p)

                ## Use the full old normal forces
                # Cost_p=params.old_forces
                # Constraint_p=-np.matmul(Constraint_N,x_s)


            elif(params.compatibility=="IPM"):
                if(warm_x.shape[0]>0):
                ###Ken's second suggestion to deal with the mu*s terms
                    s_i=find_slips(Cost_N,N,p,f,f_star,contact_static_fric,verbose=False)
                    Cost_p[0::3]=contact_static_fric*s_i
            elif(params.compatibility=="old_slips" and params.slips.shape[0]==Cost_p.shape[0]/3):
                s_i=params.slips
                # test_slip = np.dot(B,  params.dt  * v )
                # Cost_p[0::3]=contact_static_fric* np.sqrt(np.power(test_slip[1::3],2) + np.power(test_slip[2::3],2))
                Cost_p[0::3]=contact_static_fric*params.slips
                # print(test_slip, Cost_p[0::3],params.slips* contact_static_fric)
            elif(params.compatibility=="decoupled_PD"):
                pass
            else:
                print("unsupoorted compatibility criteion : '", params.compatibility, "'. Choose from:\nold_normals\nIPM\nold_slips\ndecoupled_PD")
                pass


            for iter in range(params.compatibility_max_iter):
                f_NEW=QOCC_Augmented_Lagrangian(Cost_N=Cost_N,
                                                Cost_p=Cost_p,
                                                Constraint_N=Constraint_N,
                                                Constraint_p=Constraint_p,
                                                A_E=B.T, # N
                                                x_s=x_s,
                                                x0=x_guess,
                                                QP_solver=QP_solver,
                                                contact_static_fric=contact_static_fric,
                                                tolerance=params.solver["AL_QP_tolerance"],
                                                eta0=params.solver["AL_eta0"],eta_max=params.solver["AL_eta_max"],
                                                tau_eta=params.solver["AL_tau_eta"], verbose=params.solver["AL_verbose"])

                params.f_star=f_star
                # no need to conitue
                if(params.compatibility=="incremental" or params.compatibility=="old_normals" or params.compatibility=="IPM"):
                    break

                # solve for the lagrange multipliers given the primal variables
                Ft=np.sqrt(np.power(f_NEW[1::3],2)+np.power(f_NEW[2::3],2))
                # mu_N_minus_T=contact_static_fric*np.clip(f_NEW[0::3]- Ft,0,1e10)
                mu_N_minus_T=contact_static_fric*f_NEW[0::3]- Ft
                mu_star=np.zeros((3*n_contacts,n_contacts))
                for i in range(0,n_contacts):
                    ft=Ft[i]
                    # note that the the Jacobian in the normal direction is contact_static_fric[i],
                    # however, we also added the mu*s*f term in there too, whose jacobian will cancell out this term
                    if(ft>1e-10):
                        mu_star[3*i:3*i+3,i]=np.array([0*contact_static_fric[i],
                                                            -f_NEW[3*i+1]/ft,
                                                            -f_NEW[3*i+2]/ft])

                    else:
                        mu_star[3*i:3*i+3,i]=np.array([0,0,0])
                    #     ###NOTE: this part is important if we have null contact in the system;
                    #     ###      e.g. when delta>0 hence some contacts should produce 0 force in reality
                        if(f_NEW[3*i]<1e-10):
                            B[3*i:3*i+3,:]=1e6

                G=np.block([
                            [B,                                   mu_star],
                            [np.zeros((n_contacts,6*params.nb)),  np.diag(mu_N_minus_T)]
                            ])
                # print("solving lstsq Gr=q, G.shape=",G.shape)
                #
                r=np.hstack((f_NEW, np.zeros(n_contacts)))
                s_i_old=s_i

                from scipy.optimize import linprog
                c = (contact_static_fric*f_NEW[0::3]).tolist()
                print(c,f_NEW[0::3],contact_static_fric )
                A = np.atleast_2d(+1*np.eye(n_contacts)).tolist()
                b =  np.zeros(n_contacts).tolist()
                A_eq=np.atleast_2d(np.block([[B, mu_star]])).tolist()
                b_eq=np.hstack((f_NEW, np.zeros(n_contacts))).tolist()
                res = linprog(c, A_ub=A, b_ub=b,
                                #A_eq=A_eq, b_eq=b_eq
                                )
                solution=np.linalg.lstsq(G,r)[0]
                s_i=solution[-n_contacts:]
                print( res, s_i)

                print(len(c), len(A), len(b))

                print("compatibility norm : ", norm(np.matmul(G,solution)-r))

                # NOTE: the lstsq problem does not guarantee positiveness of s_i
                if(np.min(s_i)<0):
                    print("incorrect slips result from the lstsq")
                    # s_i=np.maximum(s_i,0)
                    s_i=np.abs(s_i)


                residual=norm(s_i-s_i_old)
                # print("slips(", iter, ")=", s_i)
                # print("slips_old(", iter, ")=", s_i_old)

                print("compatibility residual %.3g"% residual)
                Cost_p[0::3]=contact_static_fric*s_i
                # print("fn(", iter, ")=", f_NEW[0::3])
                # print("ft(", iter, ")=", Ft)
                # print("mu_N_minus_T(", iter, ")=", mu_N_minus_T)
                # print("slips(", iter, ")=", s_i)
                # print("Cost_p(", iter, ")=", Cost_p)

                if(params.compatibility=="old_slips"):
                    break
                if (residual<params.compatibility_tolerance):
                    break



            # print("Compatible contact forces: \n", f_NEW.reshape(f.shape[0]//3,3))

            if(params.compatibility=="incremental"):
                # print("increments:", f_NEW)
                f=f_NEW+params.old_forces
                # print("new Force:", f)
                params.old_normals=f[0::3]
            else:
                f=f_NEW
                params.slips=s_i
                params.old_normals=f[0::3]
            print("time step force variation residual= ", norm(params.old_forces-f.copy()))
            # print("F_old after solve:", params.old_forces[::3])
            # print("F after solve:",f[::3])
        params.old_forces=f.copy()#incompatible_forces.copy()#f.copy()

        # print("%s, total_solver_iter=%d"% (params.solver["type"], numIter))
        t2=time.time()
        solve_time=t2-t1
        # print(np.dot(B[0::3,:].T, f[0::3])[0:3])
        # print(np.dot(B[1::3,:].T, f[1::3])[0:3])
        # print(np.dot(B[2::3,:].T, f[2::3])[0:3])

        # print(B[0::3,:])
        F_contact = np.dot(B.T, f)
        F = np.add(F_contact, F_ext)
    else:
        f = np.zeros((0,0))
        F = F_ext

    new_a = np.dot(params.M_inv, F)
    new_v = np.add(v, params.dt * new_a)
    new_q_dot = np.zeros(7*params.nb)


    params.contact_pairs=contact_pairs
    t1=time.time()
    for i in range(params.nb):
        # Write linear velocities as is
        new_q_dot[i*7 : i*7 + 3] = new_v[i*6 : i*6 + 3]

        # Apply quaternion identity: q_dot = 0.5 * omega * q for each omega
        rot = q[i*7 + 3 : i*7 + 7]
        w = new_v[i*6 + 3: i*6 + 6]
        rot_dot = np.zeros(4)
        rot_dot[0] = -0.5 * (rot[1]*w[0] + rot[2]*w[1] + rot[3]*w[2])
        rot_dot[1] = 0.5 * (rot[0]*w[0] - rot[2]*w[2] + rot[3]*w[1])
        rot_dot[2] = 0.5 * (rot[0]*w[1] + rot[1]*w[2] - rot[3]*w[0])
        rot_dot[3] = 0.5 * (rot[0]*w[2] - rot[1]*w[1] + rot[2]*w[0])
        new_q_dot[i*7 + 3 : i*7 + 7] = rot_dot

    new_q = q + params.dt * new_q_dot

    # NOTE for safety, normalize quaternions every step
    for i in range(params.nb):
        rot = q[i*7 + 3 : i*7 + 7]
        if abs(norm(rot, 2) - 1) > 0.01:
            q[i*7 + 3 : i*7 + 7] = rot / norm(rot, 2)
    f.shape = (n_contacts, 3)
    t2=time.time()
    update_time=t2-t1


    print("Build = %.2e, Form = %.2e, Solve = %.2e, Update = %.2e, "
        %(buildB_time,form_time,solve_time,update_time))
    return new_q, new_v, new_a, c_pos, f, B, contact_pairs, phi, totalnumIter
