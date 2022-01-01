import numpy as np
from numpy.linalg import norm

#############################################################################
### The goal of this section is to identify the set of KKT multipliers that
### represent slip. The idea is to for s_i=0 if T_i<mu* N_i else s_i=T_i/k_t
# def check_si(f,contact_static_fric,n_contacts):
#
#     Ft=np.sqrt(np.power(f[1::3],2)+np.power(f[2::3],2))
#     # print(Ft)
#     mu_N_minus_T=contact_static_fric*np.abs(f[0::3])- Ft
#     print(mu_N_minus_T)
#     OptimizeResults=optimize.linprog(-mu_N_minus_T,
#                                             A_ub=None, b_ub=None, A_eq=None, b_eq=None,
#                                             bounds=(0,None),
#                                             method='interior-point',
#                                             callback=None, options=None, x0=None,
#                                             )
#
#     print(OptimizeResults.x)

def find_slips(Cost_H,N,p,f_curr,f_star,mu, eps=1e-8, verbose=True):
    if(verbose):
        print("                    IPM solver:")
    Nc=N.shape[0]//3
    # Ne=3*Nc+1
    NI=Nc
    Nx=3*Nc
    # A_E=np.block([[N],[p]])
    A_E=N
    Ne=3*Nc

    f=f_curr.copy()
    s=1*np.ones(NI)
    z=1*np.ones(NI)
    Ft=np.sqrt(np.power(f[1::3],2)+np.power(f[2::3],2))
    # s=0.01*np.ones(NI)
    # z=0.01*np.ones(NI)
    lam=np.ones(Ne)
    fback=f.copy()

    # G=np.block([
    # [-A_E.T,                -A_I.T  ],
    # [A_E,                   np.zeros((Ne,NI)),    1e-9*np.eye(Ne),        np.zeros((Ne,NI))],
    # [A_I,                  -np.eye(NI),           np.zeros((NI,Ne)),      np.zeros((NI,NI))]
    # ])

    sigma=s@z
    # sigma=0
    for i in range(1000):
        A_I=np.zeros((NI,Nx))
        for i in range(0,NI):
            ft=np.sqrt(f[3*i+1]*f[3*i+1]+f[3*i+2]*f[3*i+2])
            if(ft>0):
                A_I[i,3*i:3*i+3]=np.array([mu[i],-f[3*i+1]/ft,-f[3*i+2]/ft])
            else:
                A_I[i,3*i:3*i+3]=np.array([mu[i],0,0])

        G=np.block([
        [Cost_H,                np.zeros((Nx,NI)),    -A_E.T,                 -A_I.T  ],
        [np.zeros((NI,Nx)),     np.diag(z),           np.zeros((NI,Ne)),      np.diag(s)],
        [A_E,                   np.zeros((Ne,NI)),    1e-9*np.eye(Ne),        np.zeros((Ne,NI))],
        [A_I,                  -np.eye(NI),           np.zeros((NI,Ne)),      np.zeros((NI,NI))]
        ])

        # G=np.block([
        # [np.zeros((Nx,NI)),    -A_E.T,                 -A_I.T  ],
        # [np.diag(z),           np.zeros((NI,Ne)),      np.diag(s)],
        # [np.eye(NI),           np.zeros((NI,Ne)),      np.zeros((NI,NI))]
        # ])

        #note that addition of this term to the KKT is necessary to ensure that
        #the slips that are found indeed are the slips in the compatible solution!!
        ## otherwise we are finding the slip for the wrong problem
        cost_p=np.zeros(Nx)
        cost_p[0::3]=mu*z
        KKT=Cost_H@f -A_E.T @ lam -A_I.T @ z + cost_p
        r_SZ=s*z-sigma*np.ones(NI)
        # r_E=np.hstack((N@(f-f_star),p@(f-f_star)))
        r_E=N@(f-f_star)

        r_I=np.zeros(NI)
        for j in range(0,NI):
            r_I[j]=mu[j]*f[3*j]-np.sqrt(f[3*j+1]*f[3*j+1] +f[3*j+2]*f[3*j+2])-s[j]
        # print(KKT.shape, r_SZ.shape, r_E.shape,r_I.shape)
        r=np.hstack((KKT,r_SZ,r_E,r_I))
        # r=np.hstack((KKT,r_SZ,r_I))
        # print(G.shape,r.shape)
        delta=-np.linalg.solve(G,r)


        alpha_1=0.9
        alpha_2=0.9

        # s+=alpha_1*delta[:NI]
        # lam+=alpha_2*delta[NI:NI+Ne]
        # z+=alpha_1*delta[NI+Ne:]


        f+=alpha_2*delta[:Nx]
        s+=alpha_1*delta[Nx:Nx+NI]
        lam+=alpha_2*delta[Nx+NI:Nx+NI+Ne]
        z+=alpha_1*delta[Nx+NI+Ne:]
        sigma*=0.9
        if(sigma<eps):
            break;
        # sigma=s@z

    if(verbose):
        print("norm(KKT):%.1e, norm(r_SZ)=%.1e, norm(r_E)=%.1e, norm(r_I)=%.1e, sigma=%.1e"
                %(norm(KKT),norm(r_SZ),norm(r_E),norm(r_I),sigma))
            # print(r_I)
        # print("s= ",s)
        print("slips= ",z)
        print("lambd= ",lam)

        # print("fback= ",f_star[::3])
        # print("fn= ",f[::3])
        print("f_ini  =\n", f_curr.reshape(Nc,3))
        print("f_end  =\n", f.reshape(Nc,3))

        # print("ft= ",f[1::3])
        print("")

    return z
        # print(A_I.shape,G.shape)
