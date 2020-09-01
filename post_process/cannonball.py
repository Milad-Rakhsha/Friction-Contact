import csv,os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from decimal import Decimal
from collections import OrderedDict
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)

path = str(sys.argv[1])
pythonpath = str(sys.argv[2])
FRAME=19
frame=str(FRAME)
frame_py=str("%03d"%FRAME)
print(frame_py)

DEM_20=path+"CannonballSMC_20particle/F_SCM_"+frame+".txt"
DEM_10=path+"CannonballSMC_10particle/F_SCM_"+frame+".txt"
DEM_5=path+ "CannonballSMC_5particle/F_SCM_"+frame+".txt"
DEM_mu0=path+"CannonballSMC_mu0/F_SCM_"+frame+".txt"

#frictionless
DVI0_mu0=path+"/"+ "0_apgd_al_0_mu0" +"/F_NSC_"+frame+".txt"
DVI1_mu0=path+"/"+ "1_apgd_al_001_mu0" +"/F_NSC_"+frame+".txt"
DVI2_mu0=path+"/"+ "2_apgd_al_01_mu0" +"/F_NSC_"+frame+".txt"
DVI3_mu0=path+"/"+ "3_apgd_al_05_mu0" +"/F_NSC_"+frame+".txt"
DVI4_mu0=path+"/"+ "4_apgd_al_1_mu0" +"/F_NSC_"+frame+".txt"
DVI5_mu0=path+"/"+ "5_apgd_al_5_mu0" +"/F_NSC_"+frame+".txt"
DVI6_mu0=path+"/"+ "6_apgd_al_10_Normal_mu0" +"/F_NSC_"+frame+".txt"


#frictional 5 particles
DVI0=path+"/"+ "0_apgd_al_0" +"/F_NSC_"+frame+".txt"
DVI1=path+"/"+ "1_apgd_al_001" +"/F_NSC_"+frame+".txt"
DVI2=path+"/"+ "2_apgd_al_01" +"/F_NSC_"+frame+".txt"
DVI3=path+"/"+ "3_apgd_al_05" +"/F_NSC_"+frame+".txt"
DVI4=path+"/"+ "4_apgd_al_1" +"/F_NSC_"+frame+".txt"
DVI5=path+"/"+ "5_apgd_al_5" +"/F_NSC_"+frame+".txt"
DVI6=path+"/"+ "6_apgd_al_10_Normal" +"/F_NSC_"+frame+".txt"


DVI0_0=path+"/"+ "0_gs_al_0" +"/F_NSC_"+frame+".txt"
DVI1_0=path+"/"+ "1_gs_al_001" +"/F_NSC_"+frame+".txt"
DVI2_0=path+"/"+ "2_gs_al_01" +"/F_NSC_"+frame+".txt"
DVI3_0=path+"/"+ "3_gs_al_05" +"/F_NSC_"+frame+".txt"
DVI4_0=path+"/"+ "4_gs_al_1" +"/F_NSC_"+frame+".txt"
DVI5_0=path+"/"+ "5_gs_al_5" +"/F_NSC_"+frame+".txt"
DVI6_0=path+"/"+ "6_gs_al_10_Normal" +"/F_NSC_"+frame+".txt"


#frictional 20 particles
DVI0_20par=path+"/"+ "0_apgd_al_0_20Particle" +"/F_NSC_"+frame+".txt"
DVI1_20par=path+"/"+ "1_apgd_al_001_20Particle" +"/F_NSC_"+frame+".txt"
DVI2_20par=path+"/"+ "2_apgd_al_01_20Particle" +"/F_NSC_"+frame+".txt"
DVI3_20par=path+"/"+ "3_apgd_al_05_20Particle" +"/F_NSC_"+frame+".txt"
DVI4_20par=path+"/"+ "4_apgd_al_1_20Particle" +"/F_NSC_"+frame+".txt"
DVI5_20par=path+"/"+ "5_apgd_al_5_20Particle" +"/F_NSC_"+frame+".txt"
DVI6_20par=path+"/"+ "6_apgd_al_10_Normal_20Particle" +"/F_NSC_"+frame+".txt"

#frictional 10 particles
DVI0_10par=path+"/"+ "0_apgd_al_0_10Particle" +"/F_NSC_"+frame+".txt"
DVI1_10par=path+"/"+ "1_apgd_al_001_10Particle" +"/F_NSC_"+frame+".txt"
DVI2_10par=path+"/"+ "2_apgd_al_01_10Particle" +"/F_NSC_"+frame+".txt"
DVI3_10par=path+"/"+ "3_apgd_al_05_10Particle" +"/F_NSC_"+frame+".txt"
DVI4_10par=path+"/"+ "4_apgd_al_1_10Particle" +"/F_NSC_"+frame+".txt"
DVI5_10par=path+"/"+ "5_apgd_al_5_10Particle" +"/F_NSC_"+frame+".txt"
DVI6_10par=path+"/"+ "6_apgd_al_10_10Particle" +"/F_NSC_"+frame+".txt"
DVI7_10par=path+"/"+ "7_apgd_al_50_10Particle" +"/F_NSC_"+frame+".txt"
DVI8_10par=path+"/"+ "8_apgd_al_100_10Particle" +"/F_NSC_"+frame+".txt"

#frictional 10 particles, Normal comp.
DVI1_10par_N=path+"/"+ "1_apgd_al_001_10Particle_N" +"/F_NSC_"+frame+".txt"
DVI2_10par_N=path+"/"+ "2_apgd_al_01_10Particle_N" +"/F_NSC_"+frame+".txt"
DVI3_10par_N=path+"/"+ "3_apgd_al_05_10Particle_N" +"/F_NSC_"+frame+".txt"
DVI4_10par_N=path+"/"+ "4_apgd_al_1_10Particle_N" +"/F_NSC_"+frame+".txt"
DVI5_10par_N=path+"/"+ "5_apgd_al_5_10Particle_N" +"/F_NSC_"+frame+".txt"
DVI6_10par_N=path+"/"+ "6_apgd_al_10_10Particle_N" +"/F_NSC_"+frame+".txt"
DVI7_10par_N=path+"/"+ "7_apgd_al_50_10Particle_N" +"/F_NSC_"+frame+".txt"
DVI8_10par_N=path+"/"+ "8_apgd_al_100_10Particle_N" +"/F_NSC_"+frame+".txt"

chrono_0=path+"/"+ "NSC_Ch" +"/F_NSC_"+frame+".txt"
chrono_1=path+"/"+ "NSC_Ch" +"/F_NSC_"+frame+".txt"


python_0= pythonpath +"cannonball/force"+frame_py+".csv"
python_1= pythonpath +"cannonball_fric/force"+frame_py+".csv"
python_2= pythonpath +"cannonball_fric_2/force"+frame_py+".csv"
python_3= pythonpath +"cannonball_fric_3/force"+frame_py+".csv"
python_4= pythonpath +"cannonball_fric_4/force"+frame_py+".csv"
python_5= pythonpath +"cannonball_fric_5/force"+frame_py+".csv"
python_6= pythonpath +"cannonball_fric_6/force"+frame_py+".csv"
python_7= pythonpath +"cannonball_fric_7/force"+frame_py+".csv"
python_8= pythonpath +"cannonball_fric_8/force"+frame_py+".csv"
python_9= pythonpath +"cannonball_fric_9/force"+frame_py+".csv"


python_30= pythonpath +"cannonball_fric_3_0/force"+frame_py+".csv"
python_40= pythonpath +"cannonball_fric_4_0/force"+frame_py+".csv"
python_50= pythonpath +"cannonball_fric_5_0/force"+frame_py+".csv"
python_60= pythonpath +"cannonball_fric_6_0/force"+frame_py+".csv"
python_70= pythonpath +"cannonball_fric_7_0/force"+frame_py+".csv"
python_80= pythonpath +"cannonball_fric_8_0/force"+frame_py+".csv"
python_90= pythonpath +"cannonball_fric_9_0/force"+frame_py+".csv"

files = OrderedDict([#
    (DEM_mu0,   {"alpha": 0, 'Np':5,      "name": "DEM_mu0",  "lineStyle": 'b-',  "markerEvery": 1, 'markersize':5, 'label': '$E=1^{10}$ Pa, $\mu= 0$'}),
    (DEM_5,     {"alpha": 0, 'Np':5,      "name": "DEM_5",  "lineStyle": 'b-',  "markerEvery": 1, 'markersize':5,   'label': '$E=1^{10}$ Pa, $\mu= 0$'}),
    (DEM_10,    {"alpha": 0, 'Np': 10,    "name": "DEM_10",  "lineStyle": 'b-',  "markerEvery": 1, 'markersize':5, 'label': '$E=1^{10}$ Pa'}),
    (DEM_20,    {"alpha": 0, 'Np': 20,    "name": "DEM_20",  "lineStyle": 'b-',  "markerEvery": 1, 'markersize':5, 'label': '$E=1^{10}$ Pa'}),

    (DVI0_mu0,   {"alpha": 0.00, 'Np':5,  "name": "DVI_APGD_al_0_mu0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI1_mu0,   {"alpha": 0.01, 'Np':5,  "name": "DVI_APGD_al_001_mu0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI2_mu0,   {"alpha": 0.1 , 'Np':5,  "name": "DVI_APGD_al_01_mu0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI3_mu0,   {"alpha": 0.5 , 'Np':5,  "name": "DVI_APGD_al_05_mu0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI4_mu0,   {"alpha": 1.0 , 'Np':5,  "name": "DVI_APGD_al_10_mu0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI5_mu0,   {"alpha": 5.0 , 'Np':5,  "name": "DVI_APGD_al_50_mu0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI6_mu0,   {"alpha": 1.0 , 'Np':5,  "name": "DVI_APGD_al_10(N)_mu0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD Normal comp.'}),

    (DVI0,   {"alpha": 0.0 , 'Np':5,    "name": "DVI_APGD_al_0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI1,   {"alpha": 0.01, 'Np':5, "name": "DVI_APGD_al_001", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI2,   {"alpha": 0.1 , 'Np':5,  "name": "DVI_APGD_al_01", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI3,   {"alpha": 0.5 , 'Np':5,  "name": "DVI_APGD_al_05", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI4,   {"alpha": 1.0 , 'Np':5,  "name": "DVI_APGD_al_10", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI5,   {"alpha": 5.0 , 'Np':5,  "name": "DVI_APGD_al_50", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI6,   {"alpha": 1.0 , 'Np':5,  "name": "DVI_APGD_al_10(N)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD Normal comp.'}),

    # (DVI0_0,   {"alpha": 0.0, 'Np':5,    "name": "DVI_GS_al_0", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'GS'}),
    # (DVI1_0,   {"alpha": 0.01, 'Np':5, "name": "DVI_GS_al_001", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'GS'}),
    # (DVI2_0,   {"alpha": 0.1, 'Np':5,  "name": "DVI_GS_al_01", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'GS'}),
    # (DVI3_0,   {"alpha": 0.5, 'Np':5,  "name": "DVI_GS_al_05", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'GS'}),
    # (DVI4_0,   {"alpha": 1.0, 'Np':5,  "name": "DVI_GS_al_10", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'GS'}),
    # (DVI5_0,   {"alpha": 5.0, 'Np':5,  "name": "DVI_GS_al_50", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'GS'}),
    # (DVI6_0,   {"alpha": 1.0, 'Np':5,  "name": "DVI_GS_al_10(N)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'GS Normal comp.'}),


    (DVI0_20par,   {"alpha": 0.00, 'Np':20, "name": "DVI_APGD_al_0_20par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI1_20par,   {"alpha": 0.01, 'Np':20, "name": "DVI_APGD_al_001_20par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI2_20par,   {"alpha": 0.1 , 'Np':20, "name": "DVI_APGD_al_01_20par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI3_20par,   {"alpha": 0.5 , 'Np':20, "name": "DVI_APGD_al_05_20par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI4_20par,   {"alpha": 1.0 , 'Np':20, "name": "DVI_APGD_al_10_20par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI5_20par,   {"alpha": 5.0 , 'Np':20, "name": "DVI_APGD_al_50_20par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI6_20par,   {"alpha": 1.0 , 'Np':20, "name": "DVI_APGD_al_10(N)_20par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD Normal comp.'}),

    (DVI0_10par,   {"alpha": 0.00, 'Np':10, "name": "DVI_APGD_al_0_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI1_10par,   {"alpha": 0.01, 'Np':10, "name": "DVI_APGD_al_001_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI2_10par,   {"alpha": 0.1 , 'Np':10, "name": "DVI_APGD_al_01_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI3_10par,   {"alpha": 0.5 , 'Np':10, "name": "DVI_APGD_al_05_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI4_10par,   {"alpha": 1.0 , 'Np':10, "name": "DVI_APGD_al_1_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI5_10par,   {"alpha": 5.0 , 'Np':10, "name": "DVI_APGD_al_5_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI6_10par,   {"alpha": 10.0 , 'Np':10, "name": "DVI_APGD_al_10_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI7_10par,   {"alpha": 50.0 , 'Np':10, "name": "DVI_APGD_al_50_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI8_10par,   {"alpha": 100.0 , 'Np':10, "name": "DVI_APGD_al_100_10par", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),

    (DVI1_10par_N,   {"alpha": 0.01, 'Np':10, "name": "DVI_APGD_al_001_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI2_10par_N,   {"alpha": 0.1 , 'Np':10, "name": "DVI_APGD_al_01_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI3_10par_N,   {"alpha": 0.5 , 'Np':10, "name": "DVI_APGD_al_05_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI4_10par_N,   {"alpha": 1.0 , 'Np':10, "name": "DVI_APGD_al_1_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI5_10par_N,   {"alpha": 5.0 , 'Np':10, "name": "DVI_APGD_al_5_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI6_10par_N,   {"alpha": 10.0 , 'Np':10, "name": "DVI_APGD_al_10_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI7_10par_N,   {"alpha": 50.0 , 'Np':10, "name": "DVI_APGD_al_50_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),
    (DVI8_10par_N,   {"alpha": 100.0 , 'Np':10, "name": "DVI_APGD_al_100_10par_N", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'APGD'}),

    # (python_0,   {"alpha": 1.0 ,  "name": "frictionless(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictionless_py'}),
    # (python_1,   {"alpha": 1.0 ,  "name": "frictional(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional'}),
    # (python_2,   {"alpha": 1.0 ,  "name": "frictional2(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional2'}),

    # (python_3,   {"alpha": 1.0 ,  "name": "frictional3(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 2000 itrs'}),
    # (python_4,   {"alpha": 1.0 ,  "name": "frictional4(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 4000 itrs'}),
    # (python_5,   {"alpha": 10.0 , "name": "frictional5(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs'}),
    # (python_6,   {"alpha": 0.01 , "name": "frictional6(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs'}),
    # (python_7,   {"alpha": 0.1 ,  "name": "frictional7(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs'}),
    # (python_8,   {"alpha": 1.0 ,  "name": "frictional8(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs'}),
    # (python_9,   {"alpha": 1.0 ,  "name": "frictional9(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs, Normal'}),

    # (python_30,   {"alpha": 1.0 ,  "name": "frictional30(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 2000 itrs, modified p'}),
    # (python_40,   {"alpha": 1.0 ,  "name": "frictional40(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 4000 itrs, modified p'}),
    # (python_50,   {"alpha": 10.0 , "name": "frictional50(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs, modified p'}),
    # (python_60,   {"alpha": 0.01 , "name": "frictional60(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs, modified p'}),
    # (python_70,   {"alpha": 0.1 ,  "name": "frictional70(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs, modified p'}),
    # (python_80,   {"alpha": 1.0 ,  "name": "frictional80(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs, modified p'}),
    # (python_90,   {"alpha": 1.0 ,  "name": "frictional90(py)", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional 1000 itrs, modified p, Normal'}),


    # (chrono_0,   {"alpha": 0.0 ,  "name": "chrono_frictional", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictional_ch'}),
    # (chrono_1,   {"alpha": 0.0 ,  "name": "chrono_frictionless", "lineStyle": 'ko-', "markerEvery": 1, 'markersize':10, 'label': 'frictionless_ch'}),

])



MARKEREVERY=1
for IDX, key in enumerate(files):

    if(IDX<=3):
        dem = pd.read_csv(key)
        dvi = pd.read_csv(key)
        ave_phi=0
    else:
        if(files[key]["Np"]==5):
            if(IDX<=10):
                dem = pd.read_csv(DEM_mu0)
            else:
                dem = pd.read_csv(DEM_5)

        elif(files[key]["Np"]==10):
                dem = pd.read_csv(DEM_10)
        elif(files[key]["Np"]==20):
                dem = pd.read_csv(DEM_20)

        dvi = pd.read_csv(key)
        ave_phi=np.mean(dvi['phi'])



    print(IDX, key,files[key]["Np"] )


    N_SMC=dem["bi"].shape[0]
    DVI_ct=np.zeros((N_SMC,4))
    for i in range(0,N_SMC):
        c_i=dem["bi"][i]
        c_j=dem["bj"][i]
        idx_list=dvi.index[ (dvi['bi'] == c_i)&(dvi['bj'] == c_j) | (dvi['bi'] == c_j)&(dvi['bj'] == c_i)  ].tolist()
        if(len(idx_list)>0):
            idx=idx_list[0]
            # print(c_i,c_j,dvi['bi'][idx],dvi['bj'][idx])
            DVI_ct[i,0]=c_i
            DVI_ct[i,1]=c_j
            DVI_ct[i,2]=dvi['Fn'][idx]
            DVI_ct[i,3]=dvi['Ft'][idx]


    fig = plt.figure(num=None,figsize=(12, 6),  dpi=100, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(121)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.plot( dem["Fn"], DVI_ct[:,2],
              "ko", markersize=2,
              label='Normal',
              )
    ax1.plot(np.sort(dem["Fn"]), np.sort(dem["Fn"]),
            "k-", linewidth=1,markersize=1, markevery=MARKEREVERY, fillstyle='none'
            )
    ax2 = fig.add_subplot(122)
    ax2.plot(dem["Ft"], DVI_ct[:,3],
              "ro", markersize=2,label='Tangential',
              )
    ax2.plot(np.sort(dem["Ft"]), np.sort(dem["Ft"]),
            "r-", linewidth=1,markersize=1, markevery=MARKEREVERY,fillstyle='none'
            )


    ax1.grid(which='both', linestyle='--', linewidth=0.5)
    # ax1.set_xlim(0, np.max(dem["Fn"]))
    # ax1.set_ylim(0, np.max(dem["Fn"]))
    ax1.legend( fancybox=True, shadow=True, ncol=1)
    ax1.grid(which='major', alpha=0.8)
    ax1.grid(which='minor', alpha=0.5)

    ax1.grid(color='k', linestyle='-', linewidth=0.2)
    ax1.set_xlabel('DEM (N)', fontsize=22)
    ax1.set_ylabel('DVI (N)', fontsize=22)


    ax2.grid(which='both', linestyle='--', linewidth=0.5)
    # ax2.set_xlim(0, np.max(dem["Ft"]))
    # ax2.set_ylim(0, np.max(dem["Ft"]))
    ax2.legend( fancybox=True, shadow=True, ncol=1)
    ax2.grid(which='major', alpha=0.8)
    ax2.grid(which='minor', alpha=0.5)
    ax2.grid(color='k', linestyle='-', linewidth=0.2)
    ax2.set_xlabel('DEM (N)', fontsize=22)
    ax2.set_ylabel('DVI (N)', fontsize=22)

    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    fig.suptitle(r'%s, $\alpha=$%s, $\bar{\phi}$=%.1e'%(files[key]["label"],files[key]["alpha"],ave_phi))
    fig.subplots_adjust(top=0.88)

    plt.savefig(str(IDX)+"_"+str(files[key]["name"])+'.png', bbox_inches='tight', fontsize=28)
    # plt.show()
    plt.close()
