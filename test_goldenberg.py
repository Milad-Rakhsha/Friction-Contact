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
# !/usr/bin/env python3
import os
# NUMTHREADS="%d"%4
# os.environ["OMP_NUM_THREADS"] = NUMTHREADS
# os.environ["OPENBLAS_NUM_THREADS"] = NUMTHREADS
# os.environ["MKL_NUM_THREADS"] = NUMTHREADS
# os.environ["VECLIB_MAXIMUM_THREADS"] = NUMTHREADS
# os.environ["NUMEXPR_NUM_THREADS"] = NUMTHREADS

import numpy as np
import sys
import os, shutil

from integrate_new import integrate, error
from IO_utils import writeParaviewFile, loadSparseMatrixFile, loadVectorFile
from writeforcefile import writeFullforcefile
from params import params

import inspect


def main(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    else:
        os.mkdir(out_dir)

    shutil.copyfile("./test_goldenberg.py", out_dir + "/input.py")
    NUM_PAR = 20
    NUM_ROW = 5
    HALF_ROW = NUM_ROW // 2

    gravity = -10
    FORCE_MG = 100
    sphere_radius = 0.01

    solver_params = {
        "type": "Jacobi",  # Gauss_Seidel, Jacobi, APGD, APGD_REF
        "GS_omega": 0.98,
        "GS_lambda": 0.98,
        "Jacobi_omega": 0.4,
        "Jacobi_lambda": 0.9,
        "max_iterations": 10000,
        "tolerance": 1e-8,

        "AL_eta0": 1e-1,
        "AL_tau_eta": 1.1,
        "AL_eta_max": 1e7,
        "AL_QP_max_iterations": 500,
        "AL_QP_tolerance": 1e-3,
        "AL_verbose": False,
        "AL_use_AugmentedLagrangian": True,
    }

    setup = {"nb": (HALF_ROW + 1) * (NUM_PAR + 1) + HALF_ROW * NUM_PAR + 3,
             "gravity": np.array([0, 0, gravity]),
             "envelope": sphere_radius * 0.1,
             "static_friction": 0.1,
             "mu_tilde": 0.5,
             "eps_0": 0.0,
             "mu_star": 0.0,
             "eps_star": 0.0,
             "tau_mu": 1.0,
             "tau_eps": 1.0,
             "dt": 5e-3,
             "time_end": 0.25,
             "prefix": out_dir + "/step",
             "suffix": ".csv",
             "unique": False,
             "Reg_T_Dir": False,
             "compatibility": "incremental",
             # "old_normals", #"old_normals", #"old_normals", #"old_slips", #None, "IPM", "decoupled_PD"
             "compatibility_tolerance": 1e-8,
             "compatibility_max_iter": 500,
             "solver": solver_params}

    model_params = params(setup)
    t_settling = 0.1
    t_pushing = 0.13
    out_fps = 100.0
    sphere_volume = 4.0 / 3.0 * np.pi * sphere_radius * sphere_radius * sphere_radius;
    sphere_mass = 1.0  # 7800 * sphere_volume;
    print("sphere_mass:", sphere_mass)
    print("nb:", setup["nb"])

    sphere_mass = 0.01
    sphere_z = -sphere_radius
    box_mass = 1.0

    bottom_left = np.array([0, 0, 0])
    x = bottom_left[0]
    z = bottom_left[2]

    rot = np.array([1, 0, 0, 0])

    id = 0
    for j in range(NUM_ROW):
        x = 0 if j % 2 == 0 else sphere_radius
        for i in range(NUM_PAR + 1 if j % 2 == 0 else NUM_PAR):
            pos = np.array([x, 0, z])
            model_params.add_sphere(pos, rot, sphere_mass, sphere_radius, id, fixed=False)
            id += 1
            x += 2 * sphere_radius
        z += np.sqrt(3) * sphere_radius

    box_hdims = np.array([sphere_radius, sphere_radius, NUM_ROW * sphere_radius])
    box_x = -sphere_radius - box_hdims[0]
    model_params.add_box(np.array([box_x, 0, box_hdims[2] - sphere_radius]), rot, box_hdims, box_mass, id, fixed=True)
    # Right wall
    id += 1
    box_x = 2 * sphere_radius * NUM_PAR + sphere_radius + box_hdims[0]
    model_params.add_box(np.array([box_x, 0, box_hdims[2] - sphere_radius]), rot, box_hdims, box_mass, id, fixed=True)
    # Bottom wall
    id += 1
    box_hdims = np.array([sphere_radius * (NUM_PAR + 3), sphere_radius, sphere_radius])
    model_params.add_box(np.array([sphere_radius * NUM_PAR, 0, -sphere_radius - box_hdims[2]]), rot, box_hdims,
                         box_mass, id, fixed=True)

    c_pos = np.array([])
    f_contact = np.array([])
    pairs = np.array([])
    gap = np.array([])
    B = np.array([])
    c_pos = np.array([])

    # print(model_params)

    step = 0
    t = 0.0
    pushing = False
    out_steps = 1.0 / (out_fps * model_params.dt)
    frame = 1

    while t <= model_params.time_end + model_params.dt:
        # if(True):
        # if step%10==1 :
        #     model_params.compatibility=None
        # else:
        #     model_params.compatibility="old_normals"

        if step % out_steps == 0:
            frame_s = '%03d' % frame
            print('t=%f, Rendering frame %s' % (t, frame_s))
            filename = model_params.prefix + frame_s + model_params.suffix
            writeParaviewFile(model_params.q, model_params.v, frame_s, model_params)
            filename = model_params.prefix + frame_s + '_forces' + model_params.suffix
            writeFullforcefile(pairs, f_contact, c_pos, B, gap, frame_s, model_params)
            frame += 1

        if t >= t_settling:
            pushing = True
            f_push = FORCE_MG * gravity * sphere_mass * (
                1 if t >= t_pushing else (t - t_settling) / (t_pushing - t_settling))
            sphere_id = HALF_ROW * (NUM_PAR + 1) + HALF_ROW * NUM_PAR + NUM_PAR // 2
            print("Pushing sphere %d at %.3f,%.3f,%.3f, with %f" % (sphere_id,
                                                                    model_params.q[sphere_id * 7 + 0],
                                                                    model_params.q[sphere_id * 7 + 1],
                                                                    model_params.q[sphere_id * 7 + 2],
                                                                    f_push))
            model_params.F_ext[6 * sphere_id + 2] = +f_push
        # print("F_old before integrate:", model_params.old_forces[::3])

        new_q, new_v, new_a, c_pos, f_contact, B, pairs, gap, numIters = \
            integrate(model_params.q, model_params.v, model_params, \
                      warm_x=f_contact, random_initial=False)
        # print("F_old after integrate:", model_params.old_forces[::3])

        print("t=%.3e, %s, total_solver_iter=%d" % (t, model_params.solver["type"], numIters))
        print("----------------------------------------------------------------")
        nc = f_contact.shape[0]
        f = f_contact.reshape(nc, 3)

        model_params.q = new_q
        model_params.v = new_v

        t += model_params.dt
        step += 1


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 2:
        print("usage: " + argv[0] + " <out_dir>")
        exit(1)

    out_dir = argv[1]
    main(out_dir)
