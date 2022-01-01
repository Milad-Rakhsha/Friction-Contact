


#!/usr/bin/env python3
import numpy as np
import sys
import os,shutil
from integrate_new import rearrange_old_values
from params import params



grav = np.array([0,0,-10])
solver_params = {
    "type": "Jacobi", #Gauss_Seidel, Jacobi, APGD, APGD_REF
    "GS_omega" : 0.98,
    "GS_lambda" :0.98,
    "Jacobi_omega" : 0.5,
    "Jacobi_lambda" :0.5,
    "max_iterations" :300,
    "tolerance" : 1e-8,
}

setup = {"nb": 2,
    "gravity" : grav,
    "envelope" : 5e-2,
    "static_friction" : 0.1,
    "mu_tilde" : 0.2,
    "eps_0" : 0.0,
    "mu_star" : 0.0,
    "eps_star" : 0.0,
    "tau_mu" : 1.0,
    "tau_eps" : 1.0,
    "dt" : 1e-2,
    "eta0": 1e-5,
    "tau_eta": 1.2,
    "eta_max": 1e5,
    "time_end": 0.5,
    "prefix" :  "step",
    "suffix" : ".csv",
    "unique" : False,
    "Reg_T_Dir": False,
    "compatibility": "old_slips",#"old_slips", #"old_slips", #None, "old_normals", "IPM", "decoupled_PD", "old_slips"
    "compatibility_tolerance": 1e-7,
    "compatibility_max_iter": 20,
    "solver": solver_params}

model_params = params(setup)



cp_new=np.array([
            [0,1],
            [0,2],
            [0, 3],
            [0, 4],
            [0, 5],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            ]
            )

cp_old=np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            ]
            )

s_i_old=np.arange(0,5)
import random
params.slips=s_i_old

params.contact_pairs=cp_old
np.random.shuffle(cp_new)
# np.random.shuffle(cp_old)


rearrange_old_values(cp_new,params)
