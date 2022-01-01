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
# Contributors: Nic Olsen
# =============================================================================
#!/usr/bin/env python3
import numpy as np

from params import params, Shape
import integrate

model_params = params()
model_params.nb = 2
model_params.hdims = np.zeros((model_params.nb, 3))
model_params.radius = 1
model_params.envelope = 0.1 * model_params.radius
model_params.shapes = [Shape.BOX, Shape.SPHERE]
model_params.static_friction = 0.5
model_params.fixed = [False, True]
model_params.alpha = 0.1
model_params.M_inv = np.zeros((6*model_params.nb, 6*model_params.nb))
model_params.dt = 1e-4
model_params.F_ext = np.array([0,0,-9.8,0,0,0, 0,0,-9.8,0,0,0])
# print(model_params)

sphere_mass = 1.0
box_mass = 4.0
box_hdims = [1,1,1]
model_params.hdims[0,:] = box_hdims

Ix = 1.0 / 12.0 * box_mass * (box_hdims[1]**2 + box_hdims[2]**2)
Iy = 1.0 / 12.0 * box_mass * (box_hdims[0]**2 + box_hdims[2]**2)
Iz = 1.0 / 12.0 * box_mass * (box_hdims[0]**2 + box_hdims[1]**2)

model_params.M_inv[0:6,0:6] = np.diag([1.0 / sphere_mass]*3 + [5.0 / (2.0*sphere_mass * model_params.radius**2)]*3)
model_params.M_inv[6:12,6:12] = np.diag([1.0 / box_mass]*3 + [1/Ix, 1/Iy, 1/Iz])

sphere_z = -1
box_z = sphere_z + model_params.radius + model_params.hdims[0,2] + 0.5 * model_params.envelope
print("sphere_z", sphere_z)
print("box_z", box_z)

q = np.zeros(7*model_params.nb)
v = np.zeros(6*model_params.nb)
# Box
q[0:7] = [0,0,box_z,1,0,0,0]

# Sphere
q[7:14] = [0,0,sphere_z,1,0,0,0]

c_pos = np.array([])
f_contact = np.array([])

result = integrate.integrate(q, v, model_params)
