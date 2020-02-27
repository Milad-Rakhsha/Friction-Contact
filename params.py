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
A class to hold data and add new bodies to the systems

"""
from enum import Enum
import numpy as np
class params:
    def __init__(self, setup):
        self.nb = setup["nb"]
        self.hdims = np.zeros((self.nb, 3), dtype='d')
        self.radius = np.zeros(self.nb, dtype='d')
        self.envelope = setup["envelope"]
        self.shapes = np.zeros(self.nb, dtype=Shape)
        self.static_friction = setup["static_friction"]
        self.fixed = np.zeros(self.nb)
        self.mu_tilde = setup["mu_tilde"]
        self.eps_0 = setup["eps_0"]
        self.mu_star = setup["mu_star" ]
        self.eps_star = setup["eps_star" ]
        self.tau_mu = setup["tau_mu"]
        self.tau_eps = setup["tau_eps"]
        self.M_inv = np.zeros((6*self.nb, 6*self.nb), dtype='d')
        self.dt = setup["dt"]
        self.time_end = setup["time_end"]
        self.F_ext = np.zeros(6*self.nb, dtype='d')
        self.prefix = setup["prefix"]
        self.suffix = setup["suffix"]
        self.objs = ['']*self.nb
        self.q = np.zeros(7*self.nb, dtype='d')
        self.v = np.zeros(6*self.nb, dtype='d')
        self.grav = setup["gravity"].copy()
        self.unique = setup["unique"]
        self.Reg_T_Dir = setup["Reg_T_Dir"]
        self.solver = setup["solver"]
        self.eta0 = setup["eta0"]
        self.tau_eta = setup["tau_eta"]
        self.eta_max = setup["eta_max"]


    def __str__(self):
        v = vars(self)
        s = '\n\n'.join("%s:\n%s" % mem for mem in v.items())
        return s

# note that hdims should be half dimension
    def add_box(self, pos, rot, hdims, mass, id, fixed=False):
        self.set_coords(pos, rot, id)
        Ix = 1.0 / 3.0 * mass * (hdims[1]**2 + hdims[2]**2)
        Iy = 1.0 / 3.0 * mass * (hdims[0]**2 + hdims[2]**2)
        Iz = 1.0 / 3.0 * mass * (hdims[0]**2 + hdims[1]**2)
        self.hdims[id,:] = hdims
        self.M_inv[6*id:6*id + 6,6*id:6*id+6] = np.diag([1.0 / mass]*3 + [1/Ix, 1/Iy, 1/Iz])
        self.shapes[id] = Shape.BOX
        self.fixed[id] = fixed
        self.F_ext[6*id : 6*id + 3] = mass * self.grav
        self.objs[id] = 'box.obj'

    def add_sphere(self, pos, rot, mass, radius, id, fixed=False):
        self.set_coords(pos, rot, id)
        self.M_inv[6*id: 6*id + 6, 6*id: 6*id + 6] = np.diag([1.0 / mass]*3 + [5.0 / (2.0*mass * radius**2)]*3)
        self.radius[id] = radius
        self.hdims[id,:] = radius
        self.shapes[id] = Shape.SPHERE
        self.fixed[id] = fixed
        self.F_ext[6*id : 6*id + 3] = mass * self.grav
        self.objs[id] = 'sphere.obj'

    def set_coords(self, pos, rot, id):
        self.q[7*id : 7*id + 3] = pos
        self.q[7*id + 3 : 7*id + 7] = rot

class Shape(Enum):
    NULL = 0
    SPHERE = 1
    BOX = 2
