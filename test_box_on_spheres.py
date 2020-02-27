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
#!/usr/bin/env python3
import numpy as np
import sys
import os,shutil

from integrate import integrate, error
from IO_utils import writeParaviewFile,loadSparseMatrixFile,loadVectorFile
from writeforcefile import writeforcefile_with_pairs
from params import params

import inspect


def main(out_dir, push_fraction, unique, addMiddle):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    else:
        os.mkdir(out_dir)

    shutil.copyfile("./test_box_on_spheres.py", out_dir+"/input.py")

    grav = np.array([0,0,-10.0])
    solver_params = {
        "type": "Jacobi", #Gauss_Seidel, Jacobi, APGD, APGD_REF
        "GS_omega" : 0.98,
        "GS_lambda" :0.98,
        "Jacobi_omega" : 0.2,
        "Jacobi_lambda" :0.2,
        "max_iterations" : 1000,
        "tolerance" : 1e-10,
    }

    setup = {"nb": 5+addMiddle,
        "gravity" : grav,
        "envelope" : 5e-3,
        "static_friction" : 0.5,
        "mu_tilde" : 0.2,
        "eps_0" : 0.0,
        "mu_star" : 0.0,
        "eps_star" : 0.0,
        "tau_mu" : 1.0,
        "tau_eps" : 1.0,
        "dt" : 1e-2,
        "eta0": 1e-3,
        "tau_eta": 2.0,
        "eta_max": 1e6,
        "time_end": 0.5,
        "prefix" : out_dir + "/step",
        "suffix" : ".csv",
        "unique" : unique,
        "Reg_T_Dir": False,
        "solver": solver_params}

    gran_params = params(setup)

    sphere_mass = 1.0
    sphere_radius = 0.1
    sphere_z=-sphere_radius
    box_mass = 1.0
    box_hdims = np.array([2,2,0.2])

    box_id = 0
    box_z = sphere_z + sphere_radius + box_hdims[2] + 0.05 * sphere_radius
    print(box_z,sphere_z)
    pos = np.array([0,0,box_z])
    theta=np.pi/8*0
    rot = np.array([np.cos(theta/2), 0,np.sin(theta/2),0])
    gran_params.add_box(pos, rot, box_hdims, box_mass, box_id)
    print("Adding box")
    print("pos", pos)
    print("hdims", box_hdims)

    id = 1
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            if ((x * y == 0) and (x + y != 0) and addMiddle):
                print ("what? ", x, y)
                continue
            pos = np.array([x,y,sphere_z])
            rot = np.array([1,0,0,0])
            gran_params.add_sphere(pos, rot, sphere_mass, sphere_radius, id, fixed=True)
            print("Adding sphere")
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
    t_settling = gran_params.time_end*0.5
    pushing = False

    max_fric = box_mass * np.abs(np.linalg.norm(grav)) * gran_params.static_friction
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
            f=f_push*(t-t_settling)/gran_params.time_end*2
            #f=f_push
            gran_params.F_ext[6*box_id + 0] = +f
            print("Pushing with %f" % f)
#            gran_params.F_ext[6*box_id + 4] = -f_push*box_hdims[2]


        new_q, new_v, new_a, c_pos, f_contact, B, pairs, gap, numIters= \
        integrate(gran_params.q, gran_params.v, gran_params,\
                  warm_x=f_contact, random_initial=False)
        print("t=%.3e, %s, total_solver_iter=%d"% (t, gran_params.solver["type"], numIters))
        nc=f_contact.shape[0]
        f=f_contact.reshape(nc,3)

        if(f.shape[0]>0):
            e=error(f[2,0], (f[0,0]+f[1,0]+f[3,0]+f[4,0])/4)
            print("Normal compatibility check: error(f_middle, (f_righ+f_left)/4)= %.3e"
                %(e)
                )

            import math
            if(not math.isnan(e)):
                err=np.append(err,e)
        # print(np.sum(f_contact,0))
        print("----------------------------------------------------------------")

        gran_params.q = new_q
        gran_params.v = new_v

        t += gran_params.dt
        step += 1

    print("error:", err)
    import matplotlib
    #matplotlib.use('TkAgg')
    matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    # matplotlib.rcParams.update({'font.size': 18})

    n=err.shape[0]
    tt=np.arange(0,gran_params.time_end-gran_params.dt,gran_params.dt )
    print("n is", n, tt)
    fig = plt.figure(num=None,  facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
    ax1.plot(tt, err)
    ax1.set_ylabel(r'normal comp err %')
    plt.savefig("normal_error")
    plt.show()

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 4:
        print("usage: " + argv[0] + " <out_dir> <push_fraction> <unique?>")
        exit(1)

    out_dir = argv[1]
    push_fraction = float(argv[2])
    unique = bool(int(argv[3]))
    main(out_dir,push_fraction, unique,1)
