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
import os

from integrate import integrate
from writefile import writeosprayfile,writeParaviewFile
from writeforcefile import writeforcefile

from params import params

def main(out_dir, push_fraction, unique):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    grav = np.array([0,0,-9.8])
    setup = {"nb": 5,
        "gravity" : grav,
        "envelope" : 1e-7,
        "static_friction" : 0.5,
        "mu_tilde" : 1.0,
        "eps_0" : 0.1,
        "mu_star" : 1e-9,
        "eps_star" : 1e-6,
        "solver": "APGD", #Gauss_Seidel, APGD, Jacobi
        "tau_mu" : 0.8,
        "tau_eps" : 0.5,
        "tolerance" : 1e-5,
        "dt" : 1e-3,
        "time_end": 3,
        "max_iterations" : 5000,
        "GS_omega" : 0.95,
        "GS_lambda" :0.95,
        "Jacobi_omega" : 0.2,
        "Jacobi_lambda" :0.9,
        "unique" : True, # Starts True for settling phase
        "prefix" : out_dir + "/step",
        "suffix" : ".csv"}

    model_params = params(setup)

    sphere_mass = 1.0
    sphere_z = -1
    sphere_radius = 1
    box_mass = 4.0
    box_hdims = np.array([2,2,0.25])

    box_id = 0
    box_z = sphere_z + sphere_radius + box_hdims[2] + 0.5 * model_params.envelope
    pos = np.array([0,0,box_z])
    rot = np.array([1,0,0,0])
    model_params.add_box(pos, rot, box_hdims, box_mass, box_id)

    id = 1
    for x in [-1,1]:
        for y in [-1,1]:
            pos = np.array([x,y,sphere_z])
            rot = np.array([1,0,0,0])
            model_params.add_sphere(pos, rot, sphere_mass, sphere_radius, id, fixed=True)
            id += 1

    c_pos = np.array([])
    f_contact = np.array([])

    step = 0
    t = 0.0
    t_settling = 0.2
    repeated_step = 0
    n_repeated_steps = 10
    contact_forces = []
    np.random.seed(0)
    pushing = False
    max_fric = box_mass * np.abs(np.linalg.norm(grav)) * model_params.static_friction
    f_push = push_fraction * max_fric

    out_fps = 100.0
    out_steps = 1.0 / (out_fps * model_params.dt)
    frame = 0
    while t < model_params.time_end:
        if step % out_steps == 0:
            frame_s = '%06d' % frame
            print("t= %f, rendering frame %s \n" %(t, frame_s) )
            filename = model_params.prefix + frame_s + model_params.suffix
            writeosprayfile(model_params.q, model_params.v, frame_s, model_params)
            writeParaviewFile(model_params.q, model_params.v, frame_s, model_params)

            frame += 1

        if pushing == False and t >= t_settling/4:
            print("Pushing with %f" % f_push)
            print("slip") if f_push > max_fric else print("stick")
            pushing = True
            model_params.F_ext[6*box_id + 0] = f_push


        new_q, new_v, new_a, c_pos, f_contact = integrate(model_params.q, model_params.v, model_params, random_initial=True)

        if t <= t_settling:
            model_params.q = new_q
            model_params.v = new_v
        else:
            model_params.unique = unique
            print("applying tikhonov regularization : ",unique)
            contact_forces.append(f_contact)
            repeated_step += 1
        if repeated_step == n_repeated_steps:
            break

        t += model_params.dt
        step += 1

    for i,f in enumerate(contact_forces):
        print("Solution {}:".format(i))
        print(f)
        print()

    diff = []
    for i,f1 in enumerate(contact_forces):
        for j,f2 in enumerate(contact_forces):
            if i <= j:
                continue
            d = np.linalg.norm(f1 - f2)
            diff.append(d)
            # print("i,j: {}".format(d))

    diff = np.array(diff)
    print("Average difference among {} solutions: {}".format(n_repeated_steps, np.average(diff)))


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 4:
        print("usage: " + argv[0] + " <out_dir> <push_fraction> <unique?>")
        exit(1)

    out_dir = argv[1]
    push_fraction = float(argv[2])
    unique = bool(int(argv[3]))
    print ("unique solution?", unique)
    main(out_dir, push_fraction, unique)
