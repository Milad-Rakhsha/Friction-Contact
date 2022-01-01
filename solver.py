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
from cvxopt import matrix

'''
Solves:
    minimize 0.5 y'Ny + y'p
    f \in K
    subject to y = B'f
    such that y has minimum W norm

Assumtions:
    - N is a diagonal matrix
    - Some actual info about K
'''
def solver(N, q, B):
    f = matrix(0.0, (B.size[0], 1))
    return f
