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

from integrate_new import integrate, error
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

    shutil.copyfile("./test_single_box.py", out_dir+"/input.py")

    grav = np.array([0,0,-10.0])
    solver_params = {
        "type": "Jacobi", #Gauss_Seidel, Jacobi, APGD, APGD_REF
        "GS_omega" : 0.98,
        "GS_lambda" :0.98,
        "Jacobi_omega" : 0.4,
        "Jacobi_lambda" :0.4,
        "max_iterations" : 10000,
        "tolerance" : 1e-10,

        "AL_eta0": 1,
        "AL_tau_eta": 1.2,
        "AL_eta_max": 1e4,
        "AL_QP_max_iterations": 100,
        "AL_QP_tolerance": 0.001,
        "AL_verbose": False,
        "AL_use_AugmentedLagrangian": True,
    }

    setup = {"nb": 2,
        "gravity" : grav,
        "envelope" : 2e-1,
        "static_friction" : 0.5,
        "mu_tilde" : 0.2,
        "eps_0" : 0.0,
        "mu_star" : 0.0,
        "eps_star" : 0.0,
        "tau_mu" : 1.0,
        "tau_eps" : 1.0,
        "dt" : 1e-2,
        "eta0": 1e-5,
        "tau_eta": 10,
        "eta_max": 1e6,
        "time_end": 0.5,
        "prefix" : out_dir + "/step",
        "suffix" : ".csv",
        "unique" : unique,
        "Reg_T_Dir": False,
        "compatibility": "incremental", # #"old_normals", #"old_normals", #"old_slips", #None, "old_normals", "IPM", "decoupled_PD", "old_slips"
        "compatibility_tolerance": 1e-8,
        "compatibility_max_iter": 500,
        "solver": solver_params}

    model_params = params(setup)
    sphere_radius = 1.0
    sphere_mass = 1.0
    box_mass = 1.0

    id = 0
    box_z =  -1
    pos = np.array([0,0,box_z])
    theta=np.pi/8*0
    rot = np.array([np.cos(theta/2), 0,np.sin(theta/2),0])
    model_params.add_box(pos, rot, np.array([10,10,0.1]), box_mass, id, fixed=True)
    print("Adding boxes")
    print("pos", pos)
    id+=1
    pos = np.array([0,0,0.2])
    rot = np.array([1,0,0,0])
    model_params.add_sphere(pos, rot, sphere_mass, sphere_radius, id, fixed=False)
    print("Adding sphere")
    print("pos", pos)
    print("radius", sphere_radius)

    c_pos = np.array([])
    f_contact = np.array([])
    pairs = np.array([])
    gap = np.array([])

    print(model_params)

    step = 0
    t = 0.0
    t_settling = model_params.time_end*0.5
    pushing = False

    max_fric = sphere_mass * np.abs(np.linalg.norm(grav)) * model_params.static_friction
    f_push = push_fraction * max_fric
    out_fps = 100.0
    out_steps = 1.0 / (out_fps * model_params.dt)
    frame = 1
    B = np.array([])
    c_pos = np.array([])
    err=np.array([])
    slips=np.array([])
    slips_t=np.array([])
    slips_ft=np.array([])

    velocity = np.array([])

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
            f=f_push
            model_params.F_ext[6*id + 0] = +f
            print(model_params.F_ext)
            print("Pushing with %f" % f)
            model_params.F_ext[6*id + 4] = -f*sphere_radius


        new_q, new_v, new_a, c_pos, f_contact, B, pairs, gap, numIters= \
        integrate(model_params.q, model_params.v, model_params,\
                  warm_x=f_contact, random_initial=False)
        print("tst sum fn=",np.sum(f_contact[:,0]))
        print("tst sum ft1=",np.sum(f_contact[:,1]))
        print("tst sum ft2=",np.sum(f_contact[:,2]))
        if(velocity.shape[0]>0):
            velocity=np.append(velocity,new_v[0:3].reshape(3,1),axis=1)
        else:
            velocity=new_v[0:3].reshape(3,1)



        print("t=%.3e, %s, total_solver_iter=%d"% (t, model_params.solver["type"], numIters))
        nc=f_contact.shape[0]
        f=f_contact.reshape(nc,3)


        if(model_params.slips.shape[0]>0):
            shape=model_params.slips.shape[0]
            ft=np.sqrt(np.power(f[:,1],2)+np.power(f[:,2],2))

            ft_p_slip=model_params.slips+ft
            if(slips.shape[0]>0):
                slips=np.append(slips,model_params.slips.reshape(shape,1),axis=1)
                slips_ft=np.append(slips_ft,ft_p_slip.reshape(shape,1),axis=1)
            else:
                slips=model_params.slips.reshape(shape,1)
                slips_ft=ft_p_slip.reshape(shape,1)

            slips_t=np.append(slips_t,t)

            print("slips:", model_params.slips.reshape(shape,1))

            print(slips_ft.shape)

        # print(np.sum(f_contact,0))
        print("----------------------------------------------------------------")

        model_params.q = new_q
        model_params.v = new_v

        t += model_params.dt
        step += 1

    print("error:", err)
    import matplotlib
    #matplotlib.use('TkAgg')
    matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams.update({'font.size': 16})

    n=err.shape[0]
    tt=np.arange(0,model_params.time_end+model_params.dt, model_params.dt )
    print("n is", n, tt)
    fig = plt.figure(num=None, figsize=(8, 5), facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
    ax1.plot(tt[-n:], err)
    ax1.set_ylabel(r'normal comp err %')
    plt.savefig(out_dir+"/normal_error.png")
    plt.show()

    color=['bo','b-','r-o','ko','k-']
    fig = plt.figure(num=None, figsize=(10, 15), facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(313)
    # ax3 = fig.add_subplot(313)
    # fig.subplots_adjust(hspace=2.0)
    size=slips.shape[1]


    ax2.plot(tt[-velocity.shape[1]:], velocity[0,], 'b-o', label="box $u_x$")
    # ax2.plot(tt[-velocity.shape[1]:], velocity[1,], 'r-o', label="box $u_y$")
    ax2.set_ylabel(r'velocities')
    # ax2.set_yscale('log')

    ax2.legend()


    ax1.plot(slips_t, slips[0,], color[0], label="contact 1")
    ax1.plot(slips_t, slips[1,], color[1], label="contact 2")
    ax1.plot(slips_t, slips[2,], color[2], label="contact 3")
    ax1.plot(slips_t, slips[3,], color[3], label="contact 4")
    ax1.plot(slips_t, slips[4,], color[4], label="contact 5")
    ax1.set_ylabel(r'slips')
    ax1.legend()
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 0.5, 0.1)
    minor_ticks = np.arange(0, 0.5, 0.02)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)



    ax3 = fig.add_subplot(312)
    # ax3 = fig.add_subplot(313)
    # fig.subplots_adjust(hspace=2.0)
    size=slips.shape[1]
    ax3.plot(slips_t, slips_ft[0,:], color[0], label="contact 1")
    ax3.plot(slips_t, slips_ft[1,:], color[1], label="contact 2")
    ax3.plot(slips_t, slips_ft[2,:], color[2], label="contact 3")
    ax3.plot(slips_t, slips_ft[3,:], color[3], label="contact 4")
    ax3.plot(slips_t, slips_ft[4,:], color[4], label="contact 5")
    ax3.set_ylabel(r'$s^i+\frac{|f^i_t|}{k}$')
    ax2.set_xlabel(r'time(s)')
    ax3.legend()

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 0.5, 0.1)
    minor_ticks = np.arange(0, 0.5, 0.02)
    ax3.set_xticks(major_ticks)
    ax3.set_xticks(minor_ticks, minor=True)
    ax3.grid(which='minor', alpha=0.2)
    ax3.grid(which='major', alpha=0.5)

    plt.savefig(out_dir+"/slips.png")
    plt.show()
if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 3:
        print("usage: " + argv[0] + " <out_dir> <push_fraction>")
        exit(1)

    out_dir = argv[1]
    push_fraction = float(argv[2])
    main(out_dir,push_fraction, False , 1)
