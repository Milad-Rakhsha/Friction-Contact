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

from integrate import integrate
from writefile import writeosprayfile
from writeforcefile import writeforcefile

from params import params

def main():
    grav = np.array([0,0,-9.8])
    setup = {"nb": 8*61+7*60, "gravity" : grav,
        "envelope" : 1e-7,
        "static_friction" : 0.5,
        "mu_tilde" : 0.1,
        "eps_0" : 0.1,
        "mu_star" : 1e-9,
        "eps_star" : 1e-6,
        "tau_mu" : 0.2,
        "tau_eps" : 0.1,
        "tolerance" : 1e-6,
        "max_iterations" : 100,
        "dt" : 1e-4,
        "time_end": 10,
        "prefix" : "data/step",
        "suffix" : ".csv"}

    # TODO eps, mu, and tau values

    gran_params = params(setup)

    sphere_mass = 1.0
    sphere_z = -1
    sphere_radius = 1
    box_mass = 4.0
    box_hdims = np.array([2,2,0.25])

    bottom_left = np.array([0,0,0])
    x = bottom_left[0]
    z = bottom_left[2]
    id = 0
    for j in range(15):
        x = 0 if j % 2 == 0 else sphere_radius
        for i in range(61 if j % 2 == 0 else 60):
            pos = np.array([x, 0, z])
            rot = np.array([1,0,0,0])
            gran_params.add_sphere(pos, rot, sphere_mass, sphere_radius, id, fixed=False)
            id += 1
            x += 2 * sphere_radius
        z += np.sqrt(3) * sphere_radius

    # pos = bottom_left + np.array([,0,-sphere_radius])
    # rot = np.array([1,0,0,0])
    # gran_params.add_box(pos, rot, box_hdims, box_mass, box_id)

    c_pos = np.array([])
    f_contact = np.array([])

    print(gran_params)

    step = 0
    t = 0.0
    t_settling = 0.1
    pushing = False

    max_fric = box_mass * np.abs(np.linalg.norm(grav)) * gran_params.static_friction
    f_push = 0.5 * max_fric
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
            writeforcefile(c_pos, f_contact, filename, gran_params)
            frame += 1

        if pushing == False and t >= t_settling:
            print("Pushing with %f" % f_push)
            print("slip") if f_push > max_fric else print("stick")
            pushing = True
            gran_params.F_ext[6*box_id + 0] = f_push

        new_q, new_v, new_a, c_pos, f_contact = integrate(gran_params.q, gran_params.v, gran_params)

        gran_params.q = new_q
        gran_params.v = new_v

        t += gran_params.dt
        step += 1

if __name__ == '__main__':
    main()
