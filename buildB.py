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
from numpy.linalg import norm
import box_sphere as bs
import sphere_sphere as ss
from params import Shape


def buildB(q, params):
    # Count contacts
    n_contacts = 0
    contact_pair = np.array([[0, 0]])
    for i in range(params.nb):
        posA = q[7 * i: 7 * i + 3]
        rotA = q[7 * i + 3: 7 * i + 7]
        shapeA = params.shapes[i]
        fixedA = params.fixed[i]
        for j in range(i + 1, params.nb):
            posB = q[7 * j: 7 * j + 3]
            rotB = q[7 * j + 3: 7 * j + 7]
            shapeB = params.shapes[j]
            fixedB = params.fixed[j]

            if fixedA and fixedB:
                continue

            if shapeA == Shape.BOX and shapeB == Shape.SPHERE:
                [collide, depth, eff_radius, n, pt1, pt2] = \
                    bs.box_sphere(posA, rotA, params.hdims[i], posB, params.radius[j], params.envelope)
            elif shapeA == Shape.SPHERE and shapeB == Shape.BOX:
                [collide, depth, eff_radius, n, pt1, pt2] = \
                    bs.box_sphere(posB, rotB, params.hdims[j], posA, params.radius[i], params.envelope)
            elif shapeA == Shape.SPHERE and shapeB == Shape.SPHERE:
                [collide, depth, eff_radius, n, pt1, pt2] = \
                    ss.sphere_sphere(posA, params.radius[i], posB, params.radius[j], params.envelope)
            else:
                print("Unimplemented shape pair: ", shapeA, shapeB)
                exit(1)

            if collide:
                contact_pair = np.append(contact_pair, [[i, j]], axis=0)
                n_contacts += 1
    contact_pair = np.delete(contact_pair, 0, axis=0)

    phi = np.zeros(n_contacts)
    B = np.zeros((3 * n_contacts, 6 * params.nb), dtype='d')  # TODO consider sparse
    c_pos = np.zeros(3 * n_contacts, dtype='d')
    contact_static_fric = np.full(n_contacts, params.static_friction,
                                  dtype='d')  # TODO need to combine - this assumes all are same
    contact_i = 0

    for i in range(params.nb):
        posA = q[7 * i: 7 * i + 3]
        rotA = q[7 * i + 3: 7 * i + 7]
        shapeA = params.shapes[i]
        fixedA = params.fixed[i]
        for j in range(i + 1, params.nb):
            posB = q[7 * j: 7 * j + 3]
            rotB = q[7 * j + 3: 7 * j + 7]
            shapeB = params.shapes[j]
            fixedB = params.fixed[j]

            if fixedA and fixedB:
                continue

            if shapeA == Shape.BOX and shapeB == Shape.SPHERE:
                [collide, depth, eff_radius, n, pt1, pt2] = \
                    bs.box_sphere(posA, rotA, params.hdims[i], posB, params.radius[j], params.envelope)
            elif shapeA == Shape.SPHERE and shapeB == Shape.BOX:
                [collide, depth, eff_radius, n, pt1, pt2] = \
                    bs.box_sphere(posB, rotB, params.hdims[j], posA, params.radius[i], params.envelope)
                # Returned pt1 on box, pt2 on sphere
                tmp = pt1
                pt1 = pt2
                pt2 = tmp

                # Returned n goes from box to sphere
                n = -n
            elif shapeA == Shape.SPHERE and shapeB == Shape.SPHERE:
                [collide, depth, eff_radius, n, pt1, pt2] = \
                    ss.sphere_sphere(posA, params.radius[i], posB, params.radius[j], params.envelope)
            else:
                print("Unimplemented shape pair: ", shapeA, shapeB)
                exit(1)

            # pt1 on i, pt2 on j, n from i to j
            if collide:
                # TODO only fill in active contacts into c_pos and clear between
                phi[contact_i] = depth

                # Generate contact basis
                nabs = np.abs(n)
                min_i = 0
                if nabs[1] < nabs[0]:
                    min_i = 1
                if nabs[2] < nabs[0] and nabs[2] < nabs[1]:
                    min_i = 2

                e_i = np.array([1.0 if k == min_i else 0.0 for k in range(3)])
                u = np.cross(e_i, n)
                w = np.cross(n, u)

                u = u / norm(u)
                w = w / norm(w)

                # B_{i,n}
                contact_pt = 0.5 * np.add(pt1, pt2)
                s_a = np.subtract(contact_pt, posA)
                s_b = np.subtract(contact_pt, posB)
                a_n = np.cross(n, s_a)
                b_n = np.cross(s_b, n)

                if not params.fixed[i]:
                    B[3 * contact_i, 6 * i: 6 * i + 3] = -n
                    B[3 * contact_i, 6 * i + 3: 6 * i + 6] = a_n
                if not params.fixed[j]:
                    B[3 * contact_i, 6 * j: 6 * j + 3] = n
                    B[3 * contact_i, 6 * j + 3: 6 * j + 6] = b_n

                # B_{i,u}
                a_u = np.cross(u, s_a)
                b_u = np.cross(s_b, u)
                if not params.fixed[i]:
                    B[3 * contact_i + 1, 6 * i: 6 * i + 3] = -u
                    B[3 * contact_i + 1, 6 * i + 3: 6 * i + 6] = a_u
                if not params.fixed[j]:
                    B[3 * contact_i + 1, 6 * j: 6 * j + 3] = u
                    B[3 * contact_i + 1, 6 * j + 3: 6 * j + 6] = b_u

                # B_{i,w}
                a_w = np.cross(w, s_a)
                b_w = np.cross(s_b, w)
                if not params.fixed[i]:
                    B[3 * contact_i + 2, 6 * i: 6 * i + 3] = -w
                    B[3 * contact_i + 2, 6 * i + 3: 6 * i + 6] = a_w
                if not params.fixed[j]:
                    B[3 * contact_i + 2, 6 * j: 6 * j + 3] = w
                    B[3 * contact_i + 2, 6 * j + 3: 6 * j + 6] = b_w

                c_pos[3 * contact_i: 3 * contact_i + 3] = contact_pt
                contact_i += 1

    return B, phi, n_contacts, c_pos, contact_static_fric, contact_pair
