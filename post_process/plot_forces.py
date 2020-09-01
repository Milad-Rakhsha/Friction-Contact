import csv,os,sys
import subprocess,re

import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from decimal import Decimal
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
MARKERSIZE=5


path_DVI_python = str(sys.argv[1])
label = str(sys.argv[2])


def prepare(path, prefix, suffix, prefix2, suffix2, pad):
        cmd=r'ls %s/%s* | wc -l '%(path,prefix)
        print(cmd)
        process = subprocess.check_output(cmd, shell=True)
        frame=int(process)
        dt=0.5/frame
        OUT=np.zeros((frame,17))
        for i in range(1,frame):
                if (pad):
                        i_frame="%03d"%i
                else:
                        i_frame=i

                FILE=path+"/"+ prefix +str(i_frame)+ suffix
                table = pd.read_csv(FILE)
                N_SMC=table["bi"].shape[0]
                OUT[i,0]=i*dt
                for contact in range(0,N_SMC):
                        c_i=table["bi"][contact]
                        c_j=table["bj"][contact]
                        #make sure i=0 and j!=0
                        if(c_j==0):
                                c_j=c_i
                                c_i=0
                        OUT[i,c_j*2-1]=table['Fn'][contact]
                        OUT[i,c_j*2]=table['Ft'][contact]

        return OUT

def make_highlights(ax):
        textstr = r'$F_t>0$'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.8, 0.5, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)
        ax.axvspan(0.25, 0.5, facecolor='blue', alpha=0.1)


def plot(label,DVI_F):

        fig = plt.figure(num=None,figsize=(10, 10),  facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # ax3 = fig.add_subplot(313)
        fig.subplots_adjust(hspace=2.0)
        color=['ro','bo','b','r-o','k', 'ko']
        for i in range(1,6):
                ax1.plot(DVI_F[:,0],DVI_F[:,i*2-1],
                        color[i],
                        linewidth=1, markersize=MARKERSIZE,label='contact %d'%i
                        )
                ax2.plot(DVI_F[:,0],DVI_F[:,i*2],
                        color[i],
                        linewidth=1, markersize=MARKERSIZE,label='contact %d'%i
                        )


        ax2.legend(fancybox=True, shadow=True, ncol=1)
        ax1.legend(fancybox=True, shadow=True, ncol=1)
        ax1.set_xlim(0, 0.5)
        ax1.set_ylim(0, 3)
        ax2.set_xlim(0, 0.5)
        # ax3.set_xlim(0, 0.5)
        ax2.set_ylim(0, 1.5)

        ax1.legend(loc='center left')
        ax2.legend(loc='center left')
        make_highlights(ax1)
        make_highlights(ax2)
        # make_highlights(ax3)

        ax2.set_xlabel(r'$t(s)$',fontsize=22,)
        ax1.set_ylabel(r'$F_n(N)$', fontsize=22,)
        ax2.set_ylabel(r'$F_t(N)$', fontsize=22,)
        # ax3.set_ylabel(r'$x(m)$',fontsize=22,)
        plt.tight_layout(pad=1.50)

        # ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        # ax2.set_ylabel(r'$F$')
        plt.savefig(label+'.png')
        #plt.show()


# DEM_F=prepare(path_DEM,'F_SCM_', '.txt', False)
#DVI_F_chrono=prepare(path_DVI_chrono,'F_NSC_', '.txt', 'data_', '.csv', False)
DVI_F_python=prepare(path_DVI_python,'stepforce','.csv', 'stepdata_sphere_', '.csv', True)


#plot("_chrono",DVI_F_chrono)
plot(label,DVI_F_python)
