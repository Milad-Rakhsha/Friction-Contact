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
from solvers import project

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mu = 2.0
nb = 50
bound = 100
points = 2 * bound * np.random.rand(3 * nb) - bound
friction = np.array([mu] * int(points.shape[0] / 3))

proj = project(points, friction)
proj2 = project(proj, friction)
err_norm = np.linalg.norm(proj - proj2)
print("Error norm ||P(x) - P(P(x))||", err_norm)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter([points[3*i+1] for i in range(nb)],\
    [points[3*i+2] for i in range(nb)],\
    [points[3*i] for i in range(nb)], color='r')

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter([points[3*i+1] for i in range(nb)],\
    [points[3*i+2] for i in range(nb)],\
    [points[3*i] for i in range(nb)], color='r')

ax.scatter([proj[3*i+1] for i in range(nb)],\
    [proj[3*i+2] for i in range(nb)],\
    [proj[3*i] for i in range(nb)], color='b')

x = y = np.linspace(-bound, bound, 100)
X, Y = np.meshgrid(x, y)
Z = 1.0 / mu * np.sqrt(X * X + Y * Y)
ax.plot_surface(X, Y, Z)
plt.show()
