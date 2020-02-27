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

from integrate import integrate
from IO_utils import writeParaviewFile,loadSparseMatrixFile,loadVectorFile
from writeforcefile import writeforcefile_with_pairs
from params import params
import inspect

def main(out_dir, unique):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    else:
        os.mkdir(out_dir)

    shutil.copyfile("./Cannonball.py", out_dir+"/input.py")


    Nb=0
    num_ball_x=6
    num_ball=num_ball_x

    for i in range(1,num_ball_x+1):
        Nb+=i*i

    print(Nb)
    grav = np.array([0,0,-10.0])

    solver_params = {
        "type": "APGD", #Gauss_Seidel, Jacobi, APGD, APGD_REF
        "GS_omega" : 0.98,
        "GS_lambda" :0.98,
        "Jacobi_omega" : 0.4,
        "Jacobi_lambda" :1.0,
        "max_iterations" : 2000,
        "tolerance" : 1e-10,
    }

    setup = {"nb": Nb, "gravity" : grav,
        "envelope" : 5e-3,
        "static_friction" : 0.5,
        "mu_tilde" : 1,
        "eps_0" : 1e-10,
        "mu_star" : 1e-8,
        "eps_star" : 1e-15,
        "tau_mu" : 1e-2,
        "tau_eps" : 1e-2,
        "dt" : 1e-3,
        "time_end": 0.2,
        "prefix" : out_dir + "/",
        "suffix" : ".csv",
        "unique" : unique,
        "Reg_T_Dir": Reg_T_Dir,
        "solver": solver_params}

    gran_params = params(setup)
    sphere_mass = 1.0
    sphere_radius = 0.1
    envelope = sphere_radius * 0.02;
    ballId = 0
    sphere_z=sphere_radius
    shift=0
    while (num_ball>=0):
        for x in range(0,num_ball):
            for y in range(0,num_ball):
                pos = np.array([x*sphere_radius*2+shift, y*sphere_radius*2+shift, sphere_z])
                rot = np.array([1,0,0,0])
                gran_params.add_sphere(pos, rot, sphere_mass, sphere_radius, ballId, fixed=(num_ball==num_ball_x) )
                # print("Adding sphere")
                # print("pos", pos)
                # print("radius", sphere_radius)
                ballId += 1
        print(ballId)
        num_ball-=1
        shift += sphere_radius;
        sphere_z += np.sqrt(2.0) * sphere_radius + envelope;


    c_pos = np.array([])
    f_contact = np.array([])
    pairs = np.array([])
    gap = np.array([])

    print(gran_params)

    step = 0
    t = 0.0


    out_fps = 100.0
    out_steps = 1.0 / (out_fps * gran_params.dt)
    frame = 0
    while t <= gran_params.time_end:
        if step % out_steps == 0:
            frame_s = '%03d' % frame
            print('t=%f, Rendering frame %s' %(t,frame_s))
            filename = gran_params.prefix + frame_s + gran_params.suffix
            writeParaviewFile(gran_params.q, gran_params.v, frame_s, gran_params)
            filename = gran_params.prefix + frame_s + '_forces' + gran_params.suffix
            writeforcefile_with_pairs(pairs, f_contact,gap, frame_s, gran_params)
            frame += 1


        new_q, new_v, new_a, c_pos, f_contact, B, pairs, gap, numIters= \
        integrate(gran_params.q, gran_params.v, gran_params,\
                  warm_x=f_contact, random_initial=False)
        print("t=%.3e, %s, total_solver_iter=%d"% (t, gran_params.solver["type"], numIters))
        print("----------------------------------------------------------------")

        gran_params.q = new_q
        gran_params.v = new_v

        t += gran_params.dt
        step += 1


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 4:
        print("usage: " + argv[0] + " <out_dir> <unique?> <reg T?>")
        exit(1)

    out_dir = argv[1]
    unique = bool(int(argv[2]))
    Reg_T_Dir = bool(int(argv[3]))

    print ("unique solution?", unique)
    main(out_dir, unique)
