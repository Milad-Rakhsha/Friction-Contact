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
from writefile import writeosprayfile
from writeforcefile import writeforcefile

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

from params import params

def main(out_dir, unique):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    grav = np.array([0,0,-9.8])
    setup = {"nb": 2,
        "gravity" : grav,
        "envelope" : 1e-7,
        "static_friction" : 0.5, # TODO allow combination of body friction at contact
        "mu_tilde" : 0.1,
        "eps_0" : 0.1,
        "mu_star" : 1e-9,
        "eps_star" : 1e-6,
        "tau_mu" : 0.2,
        "tau_eps" : 0.1,
        "tolerance" : 1e-6,
        "unique" : unique,
        "dt" : 1e-3,
        "time_end": 3,
        "max_iterations" : 300,
        "prefix" : out_dir + "/step",
        "suffix" : ".csv"}

    gran_params = params(setup)

    sphere_id = 0
    sphere_radius = 1.0
    sphere_mass = 1.0
    sphere_z = 1.1 * sphere_radius
    pos = np.array([0,0,sphere_z])
    rot = np.array([1,0,0,0])
    gran_params.add_sphere(pos, rot, sphere_mass, sphere_radius, sphere_id)

    box_id = 1
    box_mass = 4.0
    box_hdims = np.array([4,4,0.5])
    box_z = -0.5
    pos = np.array([0,0,box_z])
    rot = np.array([1,0,0,0])
    gran_params.add_box(pos, rot, box_hdims, box_mass, box_id, fixed=True)

    c_pos = np.array([])
    f_contact = np.array([])

    step = 0
    t = 0.0

    out_fps = 100.0
    out_steps = 1.0 / (out_fps * gran_params.dt)
    frame = 0
    while t < gran_params.time_end:
        if step % out_steps == 0:
            frame_s = '%06d' % frame
            print('Rendering frame ' + frame_s)
            filename = gran_params.prefix + frame_s + gran_params.suffix
            writeosprayfile(gran_params.q, gran_params.v, frame_s, gran_params)
            filename = gran_params.prefix + frame_s + '_forces' + gran_params.suffix
            frame += 1

        new_q, new_v, new_a, c_pos, f_contact = integrate(gran_params.q, gran_params.v, gran_params)

        gran_params.q = new_q
        gran_params.v = new_v

        t += gran_params.dt
        step += 1

if __name__ == '__main__':
    argv = sys.argv
    if len(sys.argv) != 3:
        print("usage " + argv[0] + " <out_dir> <unique?>")
        exit(1)

    out_dir = argv[1]
    unique = bool(int(argv[2]))
    main(out_dir, unique)
