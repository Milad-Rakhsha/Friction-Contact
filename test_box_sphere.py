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
import box_sphere as bs

box_pos = np.array([0,0,0])
box_rot = np.array([1,0,0,0])
box_hdims = np.array([1,1,1])
sphere_pos = np.array([0,0,1.5])
sphere_radius = 1
separation = 0
norm = np.zeros(3)
pt1 = np.zeros(3)
pt2 = np.zeros(3)

result = bs.box_sphere(box_pos, box_rot, box_hdims, sphere_pos, sphere_radius, separation)
print(result)
