import csv,os,sys,shutil
import subprocess,re


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib
import pandas as pd
from decimal import Decimal
from collections import OrderedDict
from matplotlib import cm
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
MARKERSIZE=5

path_DEM = str(sys.argv[1])
path_DVI = str(sys.argv[2])


def prepare(path, frame, size, prefix, suffix, pad):
        if(pad):
            f='%03d'%frame
        else:
            f=str(frame)

        file  =str(path+prefix+f+suffix)
        table = pd.read_csv(file)
        out=np.zeros((size*size,5))
        idx=table["i"][0:size*size]

        out[:,1]=idx%size
        out[:,0]=np.floor((idx)/size)
        out[:,2]=(table["fx"][0:size*size])
        out[:,3]=(table["fy"][0:size*size])
        out[:,4]=(table["fz"][0:size*size])

        return out

def title_and_save(fig,ax,cs,idx,sum):
        fig.colorbar(cs, ax=ax)
        if(idx>0):
            plt.title(r'%s, $\alpha=%s$, $\bar\mathbf{f}$=%.2f'%(ti,al,sum))
        else:
            plt.title(r'%s, $\bar\mathbf{f}$=%.2f'%(ti,sum))

def time_Step(idx):
        if(idx>0):
            dt=0.001
        else:
            dt=1.0
        return dt

def plot(x,y,z,idx,saveName,size):
    dt=time_Step(idx)
    z=z/dt
    xi = np.linspace(min(x), max(x), size)
    yi = np.linspace(min(y), max(y), size)
    X,Y=np.meshgrid(xi,yi)
    # print(x.shape, max(x),min(xi), xi.shape, max(xi), min(xi))
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True,figsize=(8, 6),  dpi=100, facecolor='w', edgecolor='k')
    cs =ax.imshow(zi.T, interpolation='nearest',cmap=cm.jet,origin='lower')
    # cs =ax.tricontourf(x,y,z/dt, cmap=cm.jet, antialiased=False)
    mean=np.mean(z)
    title_and_save(fig,ax,cs,idx, mean)
    plt.savefig(saveName)
    plt.close()


FRAME=96
size=3
dt=1e-3

out_dir="test_cannonball/%d/"%(size-1)

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)
else:
    os.mkdir(out_dir)

# DEM_F=prepare(path_DVI, FRAME, size, '/CannonballSMC_%dparticle/F_SCM_body_'%(size-1), '.txt', False)
# DVI_F1=prepare(path_DEM, FRAME, size, '/NSC_0_%dParticle/F_NSC_body_'%(size-1), '.txt', False)
# DVI_F2=prepare(path_DEM, FRAME, size, '/NSC_01_%dParticle/F_NSC_body_'%(size-1), '.txt', False)
# DVI_F3=prepare(path_DEM, FRAME, size, '/NSC_05_%dParticle/F_NSC_body_'%(size-1), '.txt', False)
# DVI_F4=prepare(path_DEM, FRAME, size, '/NSC_1_%dParticle/F_NSC_body_'%(size-1), '.txt', False)
# DVI_F5=prepare(path_DEM, FRAME, size, '/NSC_5_%dParticle/F_NSC_body_'%(size-1), '.txt', False)

DVI_=prepare(path_DVI, FRAME, size, '/force_body_', '.csv', True)


# inputs=[DEM_F, DVI_F1, DVI_F2, DVI_F3, DVI_F4, DVI_F5]
# alpha=[0, 0.0, 0.1, 0.5, 1.0, 5.0]
# title=['DEM','DVI', 'DVI', 'DVI', 'DVI', 'DVI']


inputs=[DVI_]
alpha=[0]
title=['DVI']


for data,ti,al in zip(inputs,title,alpha):
    idx=title.index(ti)
    plot(data[:,0],data[:,1],data[:,4], idx,out_dir+'N_%d_%s_%s.png'%(size,ti,al),size)
    plot(data[:,0],data[:,1],data[:,2], idx,out_dir+'T1_%d_%s_%s.png'%(size,ti,al),size)
    plot(data[:,0],data[:,1],data[:,3], idx,out_dir+'T2_%d_%s_%s.png'%(size,ti,al),size)
    plot(data[:,0],data[:,1],np.sqrt(np.power(data[:,2],2)+np.power(data[:,3],2)), idx,out_dir+'T_%d_%s_%s.png'%(size,ti,al),size)


    # dt=time_Step(idx)
    # fig, ax = plt.subplots(1,1, sharex=True, sharey=True,figsize=(8, 6),  dpi=100, facecolor='w', edgecolor='k')
    # cs =ax.tricontourf(data[:,0],data[:,1],data[:,4]/dt, 30, cmap=cm.jet)
    # title_and_save(idx)
    # plt.savefig(out_dir+'N_%d_%s_%s.png'%(size,ti,al))

    # dt=time_Step(idx)
    # fig, ax = plt.subplots(1,1, sharex=True, sharey=True,figsize=(8, 6),  dpi=100, facecolor='w', edgecolor='k')
    # title_and_save(idx)
    # plt.savefig(out_dir+'T1_%d_%s_%s.png'%(size,ti,al))

    # dt=time_Step(idx)
    # fig, ax = plt.subplots(1,1, sharex=True, sharey=True,figsize=(8, 6),  dpi=100, facecolor='w', edgecolor='k')
    # cs =ax.tricontourf(data[:,0],data[:,1],data[:,3]/dt, 30, cmap=cm.jet)
    # title_and_save(idx)
    # plt.savefig(out_dir+'T2_%d_%s_%s.png'%(size,ti,al))

    # dt=time_Step(idx)
    # fig, ax = plt.subplots(1,1, sharex=True, sharey=True,figsize=(8, 6),  dpi=100, facecolor='w', edgecolor='k')
    # cs =ax.tricontourf(data[:,0],data[:,1],np.sqrt(np.power(data[:,2],2)+np.power(data[:,3],2))/dt, 30, cmap=cm.jet)
    # title_and_save(idx)
    # plt.savefig(out_dir+'T_%d_%s_%s.png'%(size,ti,al))
