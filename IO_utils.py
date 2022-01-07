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
from params import Shape
import numpy as np
from box_sphere import Rotate


def writefile(q, v, frame, params):
    with open(params.prefix + 'box_' + frame + params.suffix, 'w') as boxfile:
        with open(params.prefix + 'sphere_' + frame + params.suffix, 'w') as spherefile:
            boxfile.write('x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz\n')
            spherefile.write('x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz\n')
            for i in range(params.nb):

                # # roll (x-axis rotation)
                # sinr_cosp = +2.0 * (q[0] * q[1] + q[2] * q[3])
                # cosr_cosp = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
                # roll = np.arctan2(sinr_cosp, cosr_cosp)
                #
                # # pitch (y-axis rotation)
                # sinp = +2.0 * (q[0] * q[2] - q[3] * q[1])
                # if np.abs(sinp) >= 1:
                #     pitch = np.copysign(np.pi / 2, sinp)
                # else:
                #     pitch = np.arcsin(sinp)
                #
                # # yaw (z-axis rotation)
                # siny_cosp = +2.0 * (q[0] * q[3] + q[1] * q[2])
                # cosy_cosp = +1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
                # yaw = np.arctan2(siny_cosp, cosy_cosp)
                # c = 360 / (2 * np.pi)
                # euler_ang = [c*roll, c*pitch, c*yaw]

                e1 = np.array([1, 0, 0])
                ori = Rotate(e1, q[7 * i + 3: 7 * i + 7])
                out = [str(q[7 * i + j]) for j in range(3)] + [str(ori[j]) for j in range(3)] + [str(v[6 * i + j]) for j
                                                                                                 in range(6)]
                if params.shapes[i] == Shape.SPHERE:
                    spherefile.write(','.join(out) + '\n')
                elif params.shapes[i] == Shape.BOX:
                    boxfile.write(','.join(out) + '\n')


def writeosprayfile(q, v, frame, params):
    with open(params.prefix + frame + params.suffix, 'w') as file:
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        file.write("mesh_name,dx,dy,dz,x1,x2,x3,y1,y2,y3,z1,z2,z3,sx,sy,sz\n")
        for i in range(params.nb):
            mesh_name = [params.objs[i]]

            pos = [str(q[7 * i + j]) for j in range(3)]

            # Get rotated basis
            rot = q[7 * i + 3: 7 * i + 7]
            b1 = Rotate(e1, rot)
            b2 = Rotate(e2, rot)
            b3 = Rotate(e3, rot)

            b1 = [str(b1[j]) for j in range(3)]
            b2 = [str(b2[j]) for j in range(3)]
            b3 = [str(b3[j]) for j in range(3)]

            scale = [str(params.hdims[i][j]) for j in range(3)]

            out = mesh_name + pos + b1 + b2 + b3 + scale
            file.write(','.join(out) + '\n')


def writeParaviewFile(q, v, frame, params):
    file = open(params.prefix + "data_sphere_" + frame + params.suffix, 'w')
    file.write("x,y,z,vx,vy,vz,radius\n")
    # f= open(params.prefix + "boxes." + frame + params.suffix, 'w')
    # file.write("x,y,z,vx,vy,vz,radius\n")
    numbox = 0
    for i in range(params.nb):
        if params.objs[i] == "box.obj":
            numbox += 1

    file_box = open(params.prefix + "data_box_" + frame + ".vtk", 'w')

    for i in range(params.nb):
        if params.objs[i] == "sphere.obj":
            pos = q[7 * i:7 * i + 3]
            vel = v[6 * i:6 * i + 3] + 1e-20
            rad = params.hdims[i][0]
            file.write("%e,%e,%e,%e,%e,%e,%e\n" % (pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], rad))
    file.close()

    file_box.write("# vtk DataFile Version 2.0\nUnstructured Grid Example\nASCII\nDATASET UNSTRUCTURED_GRID\n")
    file_box.write("POINTS " + str(numbox * 8) + " float\n")
    boxes = []
    for i in range(params.nb):
        if params.objs[i] == "box.obj":
            boxes.append(i)
            pos = q[7 * i:7 * i + 3]
            vel = v[6 * i:6 * i + 3] + 1e-20
            quat = q[7 * i + 3:7 * i + 7]
            # print("rotation:", quat)
            hdim = params.hdims[i]
            for k in (-1, +1):
                for j in (-1, +1):
                    for i in (-1, +1):
                        local_p = np.array([hdim[0] * i, hdim[1] * j, hdim[2] * k])
                        node_local = pos + Rotate(local_p, quat)
                        file_box.write("%f %f %f\n" % (node_local[0], node_local[1], node_local[2]))

    file_box.write("\nCELLS " + str(numbox) + " " + str(numbox * 9) + "\n")
    for i in range(len(boxes)):
        node = 8 * i
        file_box.write("8 %d %d %d %d %d %d %d %d\n" % (
        node, node + 1, node + 3, node + 2, node + 4, node + 5, node + 7, node + 6))

    file_box.write("\nCELL_TYPES " + str(numbox) + "\n")
    for i in boxes:
        file_box.write("12\n")

    file_box.write("\nPOINT_DATA " + str(numbox * 8))
    file_box.write("\nVECTORS V float\n")
    for i in range(params.nb):
        if params.objs[i] == "box.obj":
            pos = q[7 * i:7 * i + 3]
            vel = v[6 * i:6 * i + 3] + 1e-20
            omega = v[6 * i + 3:6 * i + 6]
            quat = q[7 * i + 3:7 * i + 7]
            hdim = params.hdims[i]
            for k in (-1, +1):
                for j in (-1, +1):
                    for i in (-1, +1):
                        local_p = np.array([hdim[0] * i, hdim[1] * j, hdim[2] * k])
                        node_vel = vel + np.cross(local_p, omega)
                        file_box.write("%f %f %f\n" % (node_vel[0], node_vel[1], node_vel[2]))

    file_box.close()


def loadSparseMatrixFile(filename):
    file = open(filename, 'r')
    N = [int(i) for i in file.readline().split()]
    matrix = np.zeros((N[0], N[1]))
    lines = file.readlines()[1:]
    for line in lines:
        row = [int(i) for i in line.split()[0:2]]
        matrix[row[0], row[1]] = float(line.split()[2])
    return matrix


def loadVectorFile(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    vec = np.array([float(lines[i].split()[0]) for i in range(len(lines))])
    return vec
