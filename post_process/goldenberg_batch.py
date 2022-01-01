import csv,os,sys
import subprocess,re

import matplotlib
#matplotlib.use('TkAgg')
# matplotlib.use('Agg')  # M be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

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
MARKERSIZE=3


def prepareDVI(path, prefix, suffix, prefix2, suffix2, pad, num_balls_last_row):
        cmd=r'ls %s/%s* | wc -l '%(path,prefix)
        process = subprocess.check_output(cmd, shell=True)
        frame=int(process)
        print(cmd,frame,num_balls_last_row)
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



path=str(sys.argv[1])
num_balls_last_row = int(sys.argv[2])
if(len(sys.argv)>3):
    path_DEM = str(sys.argv[3])
label = "forces"

# DVI0=path+"/goldenberg_1017984"
# DVI1=path+"/goldenberg_1017985"
# DVI2=path+"/goldenberg_1017986"
# DVI3=path+"/goldenberg_1017987"
# DVI4=path+"/goldenberg_1017988"
# DVI5=path+"/goldenberg_1017989"
# DVI6=path+"/goldenberg_1017990"
# DVI7=path+"/goldenberg_1017991"
# DVI8=path+"/goldenberg_1017992"
# DVI9=path+"/goldenberg_1017993"
# DVI10=path+"/goldenberg_1017994"
# DVI11=path+"/goldenberg_1017995"

DVI0=path+"/goldenberg_1019058"
DVI1=path+"/goldenberg_1019060"
DVI2=path+"/goldenberg_1019061"
DVI3=path+"/goldenberg_1019062"
DVI4=path+"/goldenberg_1019063"
DVI5=path+"/goldenberg_1019064"
DVI6=path+"/goldenberg_1019065"
DVI7=path+"/goldenberg_1019066"
DVI8=path+"/goldenberg_1019068"
DVI9=path+"/goldenberg_1019069"
DVI10=path+"/goldenberg_1019070"
DVI11=path+"/goldenberg_1019072"
files = OrderedDict([#
    # (DVI0,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": 'b-', "markerEvery": 1, 'markersize':5} ),
    (DVI1,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": 'c-', "markerEvery": 1, 'markersize':5} ),
    (DVI2,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": 'r-', "markerEvery": 1, 'markersize':5} ),
    (DVI3,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": 'k-', "markerEvery": 1, 'markersize':5} ),
    (DVI4,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    (DVI5,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    (DVI6,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    (DVI7,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    (DVI8,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    (DVI9,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    (DVI10,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    (DVI11,   {"FMAX": 10, "mu": 0.0, "categ": 'CD', "lineStyle": '-', "markerEvery": 1, 'markersize':5} ),
    ])


fig = plt.figure(num=None, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
fig.subplots_adjust(hspace=2.0)

for IDX, key in enumerate(files):
    import os
    grep = 'grep -w static_friction  %s/input.py | head -n 1 | awk {\'print $3\'} | sed "s/,//g"'%key
    mu=os.popen(grep).read()
    files[key]["mu"]=float(mu)

    grep = 'grep "FORCE_MG=" %s/input.py | head -n 1 | sed "s/FORCE_MG=//g" |  awk {\'print $1\'}'%key
    # print(grep)
    FMAX=os.popen(grep).read()
    files[key]["FMAX"]=float(FMAX)
    print("mu=", files[key]["mu"])
    print("F_MAX=", files[key]["FMAX"])

    if(files[key]["categ"]=="CD"):
        Force, Position =prepareDVI(key,'stepforce','.csv', 'stepdata_sphere_', '.csv', True, num_balls_last_row)
    if(files[key]["categ"]=="DEM"):
        Force, Position =prepareDEM(key,'stepforce','.csv', 'stepdata_sphere_', '.csv', True, num_balls_last_row)

    settle_idx=10
    i=50
    # i=Position.shape[0]-1
    # print("num_:", Force.shape[0])
    # print("pos :", Position[i,:]-Position[i,-1]/2)
    ax1.plot(Position[i,:]-Position[i,-1]/2,
            np.abs(Force[i,:] -Force[settle_idx,:])/(files[key]["FMAX"]*10) ,
            files[key]["lineStyle"],
            linewidth=1,
            markersize=MARKERSIZE,
            label=r'%s, $\mu$=%.1f, $f_{max}$=%d mg'%(files[key]["categ"],files[key]["mu"],files[key]["FMAX"])
            )


ax1.legend(fancybox=True, shadow=True, ncol=1)
# ax1.set_xlim(0, 0.5)
ax1.legend(loc='center left')
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
plt.savefig(os.path.join(path,"F="), dpi=300)
plt.show()
