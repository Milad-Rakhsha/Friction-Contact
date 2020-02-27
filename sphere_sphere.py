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

def sphere_sphere(pos1, radius1, pos2, radius2, separation):
    delta = pos2 - pos1
    dist = np.linalg.norm(delta)
    depth = dist - (radius1 + radius2) # TODO separation
    collide = (depth <= 0)
    eff_radius = (radius1 * radius2) / (radius1 + radius2)

    if dist == 0.0:
        print("pos1", pos1, "pos2", pos2)
    n = (1.0 / dist) * delta
    pt1 = pos1 + radius1 * n
    pt2 = pos2 - radius2 * n
    return [collide, depth, eff_radius, n, pt1, pt2]
