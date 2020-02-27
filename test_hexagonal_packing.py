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
# Contributors: Milad Rakhsha
# =============================================================================
#!/usr/bin/env python3
import numpy as np
import sys
import os,shutil

from integrate_new import integrate, error
from IO_utils import writeParaviewFile,loadSparseMatrixFile,loadVectorFile
from writeforcefile import writeforcefile_with_pairs
from params import params

import inspect


def main(out_dir, push_fraction):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    else:
        os.mkdir(out_dir)

    shutil.copyfile("./test_box_on_spheres.py", out_dir+"/input.py")

    grav = np.array([0,0,0])
    solver_params = {
        "type": "Jacobi", #Gauss_Seidel, Jacobi, APGD, APGD_REF
        "GS_omega" : 0.98,
        "GS_lambda" :0.98,
        "Jacobi_omega" : 0.2,
        "Jacobi_lambda" :0.2,
        "max_iterations" : 1000,
        "tolerance" : 1e-10,
    }

    setup = {"nb": 7,
        "gravity" : grav,
        "envelope" : 1e-2,
        "static_friction" : 0.5,
        "mu_tilde" : 0.2,
        "eps_0" : 0.0,
        "mu_star" : 0.0,
        "eps_star" : 0.0,
        "tau_mu" : 1.0,
        "tau_eps" : 1.0,
        "dt" : 1e-2,
        "eta0": 1e-5,
        "tau_eta": 2.5,
        "eta_max": 1e6,
        "time_end": 0.5,
        "prefix" : out_dir + "/step",
        "suffix" : ".csv",
        "unique" : 0,
        "Reg_T_Dir": False,
        "solver": solver_params}

    gran_params = params(setup)

    sphere_mass = 1.0
    sphere_radius = 1.0

    id = 0
    d_theta=60*np.pi/180
    rot = np.array([1,0,0,0])
    gran_params.add_sphere(np.array([0,0,0]), rot, sphere_mass, sphere_radius, id, fixed=False)
    id = 1
    for i in range(0,6,1):
            x=2*sphere_radius*np.cos(d_theta*i)
            y=2*sphere_radius*np.sin(d_theta*i)
            pos = np.array([x,y,0])
            gran_params.add_sphere(pos, rot, sphere_mass, sphere_radius, id, fixed=True)
            print("Adding sphere ", id)
            print("pos", pos)
            print("radius", sphere_radius)
            id += 1



    c_pos = np.array([])
    f_contact = np.array([])
    pairs = np.array([])
    gap = np.array([])

    print(gran_params)

    step = 0
    t = 0.0
    t_settling = gran_params.time_end*0.1
    pushing = False

    max_fric = 1
    f_push = push_fraction * max_fric
    out_fps = 100.0
    out_steps = 1.0 / (out_fps * gran_params.dt)
    frame = 1
    err=np.array([])
    while t <= gran_params.time_end+gran_params.dt:
        if step % out_steps == 0:
            frame_s = '%03d' % frame
            print('t=%f, Rendering frame %s' %(t,frame_s))
            filename = gran_params.prefix + frame_s + gran_params.suffix
            writeParaviewFile(gran_params.q, gran_params.v, frame_s, gran_params)
            filename = gran_params.prefix + frame_s + '_forces' + gran_params.suffix
            writeforcefile_with_pairs(pairs, f_contact,gap, frame_s, gran_params)
            frame += 1

        if  t >= t_settling-gran_params.dt:
            print("slip") if f_push > max_fric else print("stick")
            pushing = True
            # f=f_push*(t-t_settling)/gran_params.time_end*2
            f=f_push
            gran_params.F_ext[6* 0 + 0] = +f
            print("Pushing with %f" % f)
#            gran_params.F_ext[6*box_id + 4] = -f_push*box_hdims[2]


        new_q, new_v, new_a, c_pos, f_contact, B, pairs, gap, numIters= \
        integrate(gran_params.q, gran_params.v, gran_params,\
                  warm_x=f_contact, random_initial=False)
        print("t=%.3e, %s, total_solver_iter=%d"% (t, gran_params.solver["type"], numIters))
        print("----------------------------------------------------------------")

        gran_params.q = new_q
        gran_params.v = new_v

        t += gran_params.dt
        step += 1

    print("error:", err)
    import matplotlib
    matplotlib.use('TkAgg')
    # matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    # matplotlib.rcParams.update({'font.size': 18})

    fig = plt.figure(num=None,  facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
    ax1.plot(err)
    ax1.set_ylabel(r'normal comp err %')
    plt.savefig("normal_error")
    plt.show()

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 3:
        print("usage: " + argv[0] + " <out_dir> <push_fraction>")
        exit(1)

    out_dir = argv[1]
    push_fraction = float(argv[2])
    main(out_dir,push_fraction)
