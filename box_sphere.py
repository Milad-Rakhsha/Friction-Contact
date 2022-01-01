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

"""
The following is taken from ProjectChrono NarrowphaseR
"""
import numpy as np

def box_sphere(pos1, rot1, hdims1, pos2, radius2, separation):
    # Hardcoded edge_radius
    edge_radius = 1e-6

    # Express the sphere position in the frame of the box.
    spherePos = TransformParentToLocal(pos1, rot1, pos2)

    # Snap the sphere position to the surface of the box.
    boxPos = spherePos.copy()
    code = snap_to_box(hdims1, boxPos)

    delta = np.subtract(spherePos, boxPos)
    dist2 = np.dot(delta, delta)
    radius2_s = radius2 + separation
    if dist2 >= radius2_s * radius2_s or dist2 <= 1e-12:
        return [False, -1, -1, -1, -1, -1]

    # Generate contact information
    dist = np.sqrt(dist2)
    depth = np.subtract(dist, radius2)
    norm = Rotate(delta / dist, rot1)
    pt1 = TransformLocalToParent(pos1, rot1, boxPos)
    pt2 = pos2 - norm * radius2

    if (code != 1) & (code != 2) & (code != 4):
        eff_radius = radius2 * edge_radius / (radius2 + edge_radius)
    else:
        eff_radius = radius2

    return [True, depth, eff_radius, norm, pt1, pt2]

def snap_to_box(hdims, loc):
    code = 0
    if np.abs(loc[0]) > hdims[0]:
        code |= 1
        loc[0] = hdims[0] if loc[0] > 0 else -hdims[0]
    if np.abs(loc[1]) > hdims[1]:
        code |= 2
        loc[1] = hdims[1] if loc[1] > 0 else -hdims[1]
    if np.abs(loc[2]) > hdims[2]:
        code |= 4
        loc[2] = hdims[2] if loc[2] > 0 else -hdims[2]

    return code

def TransformParentToLocal(p, q, rp):
    return RotateT(np.subtract(rp, p), q)

def TransformLocalToParent(p, q, rl):
    return p + Rotate(rl, q)

def RotateT(v, q):
    q_prime = q.copy()
    q_prime[1:] *= -1
    return Rotate(v, q_prime)

def Rotate(v, q):
    t = 2 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)
