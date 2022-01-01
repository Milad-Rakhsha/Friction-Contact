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

from params import params, Shape
import buildB_box as bb

model_params = params()
model_params.nb = 2
model_params.hdims = np.zeros((model_params.nb, 3))
model_params.hdims[0,:] = [1,1,1]
model_params.radius = 1
model_params.envelope = 0.1 * model_params.radius
model_params.shapes = [Shape.BOX, Shape.SPHERE]

sphere_z = -1
box_z = sphere_z + model_params.radius + model_params.hdims[0,2] + 0.5 * model_params.envelope
print("sphere_z", sphere_z)
print("box_z", box_z)

q = np.zeros(7*model_params.nb)
# Box
q[0:7] = [0,0,box_z,1,0,0,0]

# Sphere
q[7:14] = [0,0,sphere_z,1,0,0,0]

c_pos = np.array([])
f_contact = np.array([])

result = bb.buildB(q, model_params)
print("B\n", result[0])
print("phi\n", result[1])
print("n_contacts\n", result[2])
print("c_pos\n", result[3])
