#!/usr/bin/env python2.7

import h5py
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.stats import gaussian_kde

'''
vb_p1=[0.19,  0.2,  0.21,  0.22,  0.23,  0.24,  0.25]
tr_p1=[0.77,   0.72,  0.64,   0.6,  0.55, 0.52, 0.54]
trer_p1=[0.03,  0.02,   0.02,  0.02, 0.02, 0.01, 0.02]

vb_p2=[0.18, 0.19,  0.2,  0.21,  0.22,  0.23,  0.24]
tr_p2 = [0.66, 0.69, 0.67, 0.65, 0.62, 0.58, 0.51 ]
trer_p2=[0.03, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04]
    
vb_p1_old=[ 0.2,  0.21,  0.22,  0.23,  0.24]
tr_p1_old = [0.73,  0.7,   0.69,  0.65,  0.63]
trer_p1_old = [0.01,  0.02,  0.01,  0.05,  0.12]

vb_p2_old=[0.19,  0.2,  0.21,  0.22]
tr_p2_old=[0.7,    0.69,  0.72,  0.68]
trer_p2_old=[0.02, 0.02,  0.02,  0.02]

vb_p1_rtcut=[0.19,  0.2,  0.21,  0.22,  0.23,  0.24,  0.25]
tr_p1_rtcut=[0.54, 0.48,  0.47, 0.44,  0.43, 0.44, 0.47]
trer_p1_rtcut=[0.03, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02]

vb_p2_rtcut=[0.18, 0.19,  0.2,  0.21,  0.22,  0.23,  0.24]
tr_p2_rtcut = [0.47,  0.49, 0.45, 0.46, 0.45, 0.45, 0.46 ]
trer_p2_rtcut=[0.02, 0.01, 0.02, 0.01, 0.02, 0.03, 0.03]
'''

vb_b1=[0.19,  0.2,  0.21,  0.22,  0.23,  0.24]
tr_b1=[0.454,   0.48,   0.44,    0.44,   0.44, 0.43]
trer_b1=[0.05,     0.015,  0.01,    0.01,  0.01,  0.01]


vb_b2=[0.2,  0.21,  0.22,  0.23,  0.24, 0.25]
tr_b2=[0.4,    0.41,   0.47,   0.47,  0.47,  0.46]
trer_b2=[0.04, 0.02,  0.02, 0.03,  0.03, 0.03]
       
vb_b3=[0.19, 0.2,  0.21,  0.22,  0.23,  0.24]
tr_b3=[0.49,  0.5,  0.46,  0.44,  0.44,   0.5]
trer_b3=[0.02, 0.01,  0.01,  0.01, 0.02, 0.07]

#vb_b4=[0.18, 0.19, 0.2]
#tr_b4=[0.5,  0.51, 0.47]
#trer_b4=[0.02, 0.02, 0.02]

vb_b4=[0.18, 0.19, 0.2, 0.21, 0.22, 0.23]
tr_b4=[0.5,  0.51, 0.47, 0.45, 0.46, 0.44]
trer_b4=[0.02, 0.02, 0.02, 0.02, 0.02, 0.03]

'''
vb_p3=[0.19,  0.2,  0.21,  0.22,  0.23,  0.24]
tr_p3=[0.63, 0.64, 0.52, 0.48, 0.47, 0.45 ]
trer_p3=[0.02, 0.02, 0.01, 0.01, 0.01, 0.01]

vb_p4=[0.19,  0.2,  0.21, 0.22]
tr_p4=[0.63, 0.56, 0.48, 0.47]
trer_p4=[0.02, 0.01, 0.01, 0.02]
'''

plt.figure()
                   
plt.errorbar(vb_b1, tr_b1, yerr=trer_b1,ls='-', fmt='ok',linewidth = 2,markersize=3, capsize=2, label="P2 (broad band)")
plt.errorbar(vb_b2, tr_b2, yerr=trer_b2,ls='-', fmt='ob',linewidth = 2,markersize=3, capsize=2, label="P5 (broad band)")
plt.errorbar(vb_b3, tr_b3, yerr=trer_b3,ls='-', fmt='or',linewidth = 2,markersize=3, capsize=2, label="P3 (broad band)")
plt.errorbar(vb_b4, tr_b4, yerr=trer_b4,ls='-', fmt='og',linewidth = 2,markersize=3, capsize=2, label="P4 (broad band)")
#plt.errorbar(vb_p1_rtcut, tr_p1_rtcut, yerr=trer_p1_rtcut,ls='-', fmt='ok',linewidth = 2,markersize=2, capsize=2, label="P1 (broad band w rising time cut)")
#plt.errorbar(vb_p2_rtcut, tr_p2_rtcut, yerr=trer_p2_rtcut,ls='-', fmt='ob',linewidth = 2,markersize=2, capsize=2, label="P2 (broad band w rising time cut)")
#plt.errorbar(vb_p1_old, tr_p1_old, yerr=trer_p1_old,ls='--', fmt='ok',linewidth = 2,markersize=2, capsize=2, label="P1 (narrow band)")
#plt.errorbar(vb_p1_old, tr_p1_old, yerr=trer_p1_old,ls='--', fmt='ok',linewidth = 2,markersize=2, capsize=2, label="P1 (narrow band)")
#plt.errorbar(vb_p2_old, tr_p2_old, yerr=trer_p2_old,ls='--', fmt='ob',linewidth = 2,markersize=2, capsize=2, label="P2 (narrow band)")
#plt.errorbar(vb_p3, tr_p3, yerr=trer_p3, fmt='-or',linewidth = 2,markersize=5, capsize=3, label="P3")
#plt.errorbar(vb_p4, tr_p4, yerr=trer_p4, fmt='-og',linewidth = 2,markersize=5, capsize=3, label="P4")
plt.xlabel("Bias voltage (V)")
plt.ylabel("Time resolution in sigma (ns)")
plt.ylim(0.2,0.7)
plt.legend()
plt.show()
     
