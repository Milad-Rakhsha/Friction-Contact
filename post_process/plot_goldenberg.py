import csv,os,sys
import subprocess,re

import matplotlib
#matplotlib.use('TkAgg')
# matplotlib.use('Agg')  # M be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import colors



import numpy as np
import pandas as pd
from decimal import Decimal
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
MARKERSIZE=1


path_DVI_python1 = str(sys.argv[1])
num_balls_last_row = int(sys.argv[2])
if(len(sys.argv)>3):
    path_DVI_python2 = str(sys.argv[3])
if(len(sys.argv)>4):
    path_DEM = str(sys.argv[4])

label = "forces"
f_MAX=10
settle_idx=8
settle_idx_CCD=8
num_particles=908
Radius=0.01
gravity=10
mu=0.1
dt=0.01

def prepare(path, prefix, suffix, pad, num_balls_last_row):
        cmd=r'ls %s/%s* | wc -l '%(path,prefix)
        print(cmd)
        process = subprocess.check_output(cmd, shell=True)
        frame=int(process)
        OUT_F=np.zeros((frame,num_balls_last_row))
        OUT_C=np.zeros((frame,num_balls_last_row))

        for i in range(1,frame):
                if (pad):
                        i_frame="%03d"%i
                else:
                        i_frame=i

                FILE=path+"/"+ prefix +str(i_frame)+ suffix
                table = pd.read_csv(FILE)
                bottom_wall_idx=max(np.max(table["bi"]),np.max(table["bj"]))
                nc=table["bi"].shape[0]

                for contact in range(0,nc):
                        c_i=table["bi"][contact]
                        c_j=table["bj"][contact]
                        if(table["bi"][contact]==bottom_wall_idx or table["bj"][contact]==bottom_wall_idx):
                            #make sure i=0 and j!=0
                            if(c_i==bottom_wall_idx):
                                c_i=c_j

                            OUT_F[i,c_i]=table['Fn'][contact]
                            OUT_C[i,c_i]=table['x'][contact]

        return OUT_F,OUT_C


def prepareDEM(path, num_balls_last_row):
        OUT_F=np.zeros((2,num_balls_last_row-1))
        OUT_C=np.zeros((2,num_balls_last_row-1))
        table = pd.read_csv(path+"/settling_pos_force.csv")
        OUT_F[0,:]=np.array(table.iloc[:, [1]]).flatten()
        OUT_C[0,:]=np.array(table.iloc[:, [0]]).flatten()
        table = pd.read_csv(path+"/ending_pos_force.csv")
        OUT_F[1,:]=np.array(table.iloc[:, [1]]).flatten()
        OUT_F[1,:]=np.array(table.iloc[:, [1]]).flatten()

        return OUT_F,OUT_C


def plot(label,Force_CD,Position_CD, Force_CCD=None, Position_CCD=None, ForceDEM=None, PositionDEM=None):
        fig = plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(111)
        fig.subplots_adjust(hspace=2.0)
        step=Position_CD.shape[0]-1
        print(Position_CD.shape)
        print("Force:", Force_CD.shape, "particles=", Position_CD.shape)
        num_timeSteps=Position_CD.shape[0]-settle_idx
        cmap = plt.get_cmap("viridis")

        mycmap = cmap(np.linspace(0, 1, num_timeSteps))
        norm = matplotlib.colors.Normalize(
                            vmin=np.min(0),
                            vmax=np.max(num_timeSteps*dt))
        s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        s_m.set_array([])
        for time_step in range(num_timeSteps):
            step=time_step+settle_idx
            f_gravity=  np.sum(Force_CD[settle_idx-1,:])
            f_execc=np.sum(Force_CD[step,:] -Force_CD[settle_idx-1,:])
            f_execc_max=np.max(Force_CD[step,:] -Force_CD[settle_idx-1,:])
            ax1.plot((Position_CD[step,:]-Position_CD[step,-1]/2.0)/Radius,
                    (Force_CD[step,:] -Force_CD[settle_idx-1,:])/f_MAX ,
                    # 'b--',
                    c=mycmap[time_step],
                    # cmap='viridis',
                    # linewidth=pow(time_step/num_timeSteps,1)*2,
                    markersize=MARKERSIZE,
                    # label="t=%.2f, $\Sigma f_{gravity}=%.2f, \Sigma f_{excess}=%.2f$"%(dt,f_gravity,f_execc)
                    )
        clb=fig.colorbar(s_m)
        clb.ax.tick_params(labelsize=10)
        clb.ax.set_title('Time',fontsize=10)

        ax1.plot((Position_CD[step,:]-Position_CD[step,-1]/2.0)/Radius,
                (Force_CD[step,:] -Force_CD[settle_idx-1,:])/f_MAX ,
                'k-o',
                linewidth=0.2,
                markersize=MARKERSIZE,
                label="Steady State"
                )
        print("CCD sum of setteled forces:", f_gravity, "total Weight=", num_particles*0.01*gravity)
        print("CCD sum of excess forces:", f_execc)
        print("CCD max of excess forces:", f_execc_max)

        # ax1.plot((Position_CD[step,:]-Position_CD[step,-1]/2.0)/Radius,
        #         np.abs(Force_CD[step,:])/f_MAX ,
        #         'r-o',
        #         linewidth=1,
        #         markersize=MARKERSIZE,
        #         label="Pushing Force"
        #         )
        # ax1.plot((Position_CD[settle_idx-1,:]-Position_CD[settle_idx-1,-1]/2.0)/Radius,
        #         np.abs(Force_CD[settle_idx-1,:])/f_MAX ,
        #         'k-o',
        #         linewidth=1,
        #         markersize=MARKERSIZE,
        #         label="Settled Force"
        #         )
        if(Force_CCD is not None):
            print(Position_CCD.shape)
            print(Position_CCD.shape)
            step=Position_CCD.shape[0]-1
            ax1.plot((Position_CCD[step,:]-Position_CCD[step,-1]/2.0)/Radius,
                    (Force_CCD[step,:] -Force_CCD[settle_idx_CCD-1,:])/f_MAX ,
                    'b-o',
                    linewidth=1,
                    markersize=MARKERSIZE,
                    label="Compatible CD"
                    )
            f_gravity=  np.sum(Force_CCD[settle_idx_CCD-1,:])
            f_execc=np.sum(Force_CCD[step,:] -Force_CCD[settle_idx_CCD-1,:])
            f_execc_max=np.max(Force_CCD[step,:] -Force_CCD[settle_idx_CCD-1,:])
            #
            print("CCD sum of setteled forces:", f_gravity, "total Weight=", num_particles*0.01*gravity)
            print("CCD sum of excess forces:", f_execc)
            print("CCD max of excess forces:", f_execc_max)

            t = ax1.annotate( "$\Sigma f_{gravity}=%.2f$, Weight=%.2f"%(f_gravity,num_particles*0.01*gravity), xy=(0.5, 0.05), xycoords='axes fraction', ha="center", va="center", rotation=0, size=12)
            t = ax1.annotate("$\Sigma f_{excess}=%.2f$,  $F_{ext}$=%.2f"%(f_execc,f_MAX),xy=(0.5, 0.1), xycoords='axes fraction', ha="center", va="center", rotation=0, size=12)

        if(ForceDEM is not None):
            ax1.plot((PositionDEM[0,:])/100,
                    np.abs(ForceDEM[1,:] -ForceDEM[0,:])/(f_MAX*100) ,
                    'r-o',
                    linewidth=1,
                    markersize=MARKERSIZE,
                    label="DEM"
                    )

        # Major ticks every 20, minor ticks every 5
        # major_ticks = np.arange(-num_balls_last_row+1, num_balls_last_row)
        # minor_ticks = np.arange(0, 0.5, 0.02)
        plt.title("$\mu=%0.1f$"%mu)
        ax1.legend(fancybox=True, shadow=True, ncol=1)
        ax1.set_ylim(-0.02,0.1)
        plt.legend()


        ax1.legend(loc='upper left')
        ax1.set_ylabel(r'$F_n/F_{Ext}$', fontsize=12,)
        ax1.set_xlabel(r'$X/R$', fontsize=12,)

        # ax3.set_ylabel(r'$x(m)$',fontsize=22,)
        plt.tight_layout(pad=1.50)
        # ax1.set_xticks(major_ticks)
        # ax1.set_xticks(minor_ticks, minor=True)
        # ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)
        # plt.grid()
        # ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        # ax2.set_ylabel(r'$F$')
        plt.savefig(os.path.join(path_DVI_python1,label), dpi=300)
        plt.show()



Force_CD,Position_CD=prepare(path_DVI_python1,'stepforce','.csv', True,num_balls_last_row)

if(len(sys.argv)==4):
    Force_CCD,Position_CCD=prepare(path_DVI_python2,'stepforce','.csv', True,num_balls_last_row)
    plot(label,Force_CD,Position_CD,Force_CCD,Position_CCD)
elif(len(sys.argv)==5):
    Force_CCD,Position_CCD=prepare(path_DVI_python2,'stepforce','.csv', True,num_balls_last_row)
    ForceDEM,PositionDEM=prepareDEM(path_DEM, num_balls_last_row)
    plot(label,Force_CD,Position_CD,Force_CCD,Position_CCD, ForceDEM,PositionDEM)
else:
    plot(label,Force_CD,Position_CD)
