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
Writes contact forces to files 
"""
import numpy as np
def writeforcefile(c_pos, f_contact, filename, params):
    with open(filename, 'w') as file:
        file.write('cx,cy,cz,fn,fu,fw\n')
        if len(f_contact) != 0:
            for i in range(f_contact.shape[0]):
                out = [str(c_pos[i*3 + j]) for j in range(3)] + [str(f_contact[i,j]) for j in range(3)]
                file.write(','.join(out) + '\n')
        else:
            out = [str(0.0)]*6
            file.write(','.join(out) + '\n')


def writeforcefile_with_pairs(contact_pair, f_contact, phi, frame, params):
    file= open(params.prefix  + "force" +frame + params.suffix, 'w')
    file.write('bi,bj,Fn,Ft,phi\n')
    if len(f_contact) != 0:
        for i in range(f_contact.shape[0]):
            out = [str(contact_pair[i][j]) for j in range(2)] + [str(f_contact[i,0]),str(np.linalg.norm(f_contact[i,1:2],2))] + [str(phi[i])]
            file.write(','.join(out) + '\n')
    else:
        pass
