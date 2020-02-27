#!/usr/bin/env python3
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
import numpy as np

from integrate import integrate
from writefile import writeosprayfile
from writeforcefile import writeforcefile

from params import params, Shape

grav = np.array([0,0,-9.8])
setup = {"nb": 2, "gravity" : grav, \
    "envelope" : 1e-7, \
    "static_friction" : 0.5, \
    "alpha" : 0.1, \
    "dt" : 1e-4, \
    "time_end": 10, \
    "prefix" : "data/step", \
    "suffix" : ".csv"}

gran_params = params(setup)

sphere_id = 1
sphere_pos = np.array([0,0,-1])
sphere_rot = np.array([1,0,0,0])
sphere_mass = 1.0
sphere_radius = 1

box_id = 0
box_hdims = np.array([1,1,1])
box_z = sphere_radius + box_hdims[2] + 0.5 * gran_params.envelope
box_pos = np.array([0.1,0,box_z])
box_rot = np.array([1,0,0,0])
box_mass = 4.0

gran_params.add_box(box_pos, box_rot, box_hdims, box_mass, box_id)
gran_params.add_sphere(sphere_pos, sphere_rot, sphere_mass, sphere_radius, sphere_id, fixed=True)

c_pos = np.array([])
f_contact = np.array([])

print(gran_params)

step = 0
t = 0.0
t_settling = 0.1
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

    new_q, new_v, new_a, c_pos, f_contact = integrate(gran_params.q, gran_params.v, gran_params)

    gran_params.q = new_q
    gran_params.v = new_v

    t += gran_params.dt
    step += 1
