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
        "Jacobi_omega" : 0.3,
        "Jacobi_lambda" :0.9,
        "max_iterations" : 1000,
        "tolerance" : 1e-9,

        "AL_eta0": 1e-2,
        "AL_tau_eta": 1.1,
        "AL_eta_max": 1e6,
        "AL_QP_max_iterations": 500,
        "AL_QP_tolerance": 0.001,
        "AL_verbose": False,
        "AL_use_AugmentedLagrangian": False,

    }
    sphere_radius = 1.0
    sphere_mass = 1.0

    setup = {"nb": 7,
        "gravity" : grav,
        "envelope" : sphere_radius*0.1,
        "static_friction" : 0.5,
        "mu_tilde" : 0.2,
        "eps_0" : 0.0,
        "mu_star" : 0.0,
        "eps_star" : 0.0,
        "tau_mu" : 1.0,
        "tau_eps" : 1.0,
        "dt" : 1e-2,
        "time_end": 0.5,
        "prefix" : out_dir + "/step",
        "suffix" : ".csv",
        "unique" : 0,
        "Reg_T_Dir": False,
        "compatibility": "incremental",  #None, "old_normals", "IPM", "decoupled_PD", "old_slips"
        "compatibility_tolerance": 1e-10,
        "compatibility_max_iter": 20,
        "solver": solver_params}

    model_params = params(setup)



    id = 0
    d_theta=60*np.pi/180
    rot = np.array([1,0,0,0])
    model_params.add_sphere(np.array([0,0,0]), rot, sphere_mass, sphere_radius, id, fixed=False)
    id = 1
    for i in range(0,6,1):
            x=2*sphere_radius*np.cos(d_theta*i)
            y=2*sphere_radius*np.sin(d_theta*i)
            pos = np.array([x,y,0])
            model_params.add_sphere(pos, rot, sphere_mass, sphere_radius, id, fixed=True)
            print("Adding sphere ", id)
            print("pos", pos)
            print("radius", sphere_radius)
            id += 1



    c_pos = np.array([])
    f_contact = np.array([])
    pairs = np.array([])
    gap = np.array([])
    slips=np.array([])
    slips_t=np.array([])

    err=np.array([])

    print(model_params)

    step = 0
    t = 0.0
    t_settling = model_params.time_end*0.1
    pushing = False

    max_fric = 1
    f_push = push_fraction * max_fric
    out_fps = 100.0
    out_steps = 1.0 / (out_fps * model_params.dt)
    frame = 1
    while t <= model_params.time_end+model_params.dt:

        if step % out_steps == 0:
            frame_s = '%03d' % frame
            print('t=%f, Rendering frame %s' %(t,frame_s))
            filename = model_params.prefix + frame_s + model_params.suffix
            writeParaviewFile(model_params.q, model_params.v, frame_s, model_params)
            filename = model_params.prefix + frame_s + '_forces' + model_params.suffix
            writeforcefile_with_pairs(pairs, f_contact,gap, frame_s, model_params)
            frame += 1

        if  t >= t_settling-model_params.dt:
            print("slip") if f_push > max_fric else print("stick")
            pushing = True
            # f=f_push*(t-t_settling)/model_params.time_end*2
            # f=f_push
            f=f_push*min(max(t-t_settling,0),2*model_params.dt)/(2*model_params.dt)

            model_params.F_ext[6* 0 + 0] = +f
            print("Pushing with %f" % f)
#            model_params.F_ext[6*box_id + 4] = -f_push*box_hdims[2]

        print("F_old before integrate:", model_params.old_forces[::3])

        new_q, new_v, new_a, c_pos, f_contact, B, pairs, gap, numIters= \
        integrate(model_params.q, model_params.v, model_params,\
                  warm_x=f_contact, random_initial=False)

        print("F_old after integrate:", model_params.old_forces)


        if(model_params.slips.shape[0]==3):
            shape=model_params.slips.shape[0]
            if(slips.shape[0]>0):
                slips=np.append(slips,model_params.slips.reshape(shape,1),axis=1)
            else:
                slips=model_params.slips.reshape(shape,1)

            slips_t=np.append(slips_t,t)
            print("slips:", model_params.slips.reshape(shape,1))

        nc=f_contact.shape[0]
        f=f_contact.reshape(nc,3)
        if(f.shape[0]>0 and f.shape[0]>1):
              e=error(2*0.5*f[0,0], (f[1,0]+f[-1,0]))
              print("Normal compatibility check: error(f_middle, (f_bot+f_top))= %.3e"
                  %(e)
                  )

              import math
              if(not math.isnan(e)):
                  err=np.append(err,e)
        print("t=%.3e, %s, total_solver_iter=%d"% (t, model_params.solver["type"], numIters))
        print("----------------------------------------------------------------")

        model_params.q = new_q
        model_params.v = new_v

        t += model_params.dt
        step += 1

    print("error:", err)
    import matplotlib
    matplotlib.use('TkAgg')
    # matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    # matplotlib.rcParams.update({'font.size': 18})

    fig = plt.figure(num=None, figsize=(5, 5), facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
    ax1.plot(err)
    ax1.set_yscale('log')
    ax1.set_ylabel(r'normal comp err %')
    plt.savefig(out_dir+"/normal_error")
    plt.show()

    if(setup["compatibility"]=="old_slips"):
        color=['k','r-','ro']
        fig = plt.figure(num=None, figsize=(5, 5), facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(111)
        size=slips.shape[1]
        ax1.plot(slips_t[-size:], slips[0,-size:], color[0], label="contact 1")
        ax1.plot(slips_t[-size:], slips[1,-size:], color[1], label="contact 2")
        ax1.plot(slips_t[-size:], slips[2,-size:], color[2], label="contact 6")
        ax1.set_ylabel(r'slips')
        plt.legend()
        plt.savefig(out_dir+"/slips.png")
        plt.show()

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 3:
        print("usage: " + argv[0] + " <out_dir> <push_fraction>")
        exit(1)

    out_dir = argv[1]
    push_fraction = float(argv[2])
    main(out_dir,push_fraction)
