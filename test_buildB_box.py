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

gran_params = params()
gran_params.nb = 2
gran_params.hdims = np.zeros((gran_params.nb, 3))
gran_params.hdims[0,:] = [1,1,1]
gran_params.radius = 1
gran_params.envelope = 0.1 * gran_params.radius
gran_params.shapes = [Shape.BOX, Shape.SPHERE]

sphere_z = -1
box_z = sphere_z + gran_params.radius + gran_params.hdims[0,2] + 0.5 * gran_params.envelope
print("sphere_z", sphere_z)
print("box_z", box_z)

q = np.zeros(7*gran_params.nb)
# Box
q[0:7] = [0,0,box_z,1,0,0,0]

# Sphere
q[7:14] = [0,0,sphere_z,1,0,0,0]

c_pos = np.array([])
f_contact = np.array([])

result = bb.buildB(q, gran_params)
print("B\n", result[0])
print("phi\n", result[1])
print("n_contacts\n", result[2])
print("c_pos\n", result[3])
