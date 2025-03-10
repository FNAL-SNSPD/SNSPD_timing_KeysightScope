#!/usr/bin/env python2.7

import h5py
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.stats import gaussian_kde


def find_nearest(a, maxid, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    try:
        idx = np.abs(a[:maxid] - a0).argmin()
    except:
        idx = 0
    #return a.flat[idx]
    return idx

def find_rising_time_ch2(filename, iwave):

    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch2 = waveforms["Channel 2"]

    total_wfm_n = len(waveforms_ch2)
    #print(total_wfm_n)
    iwave_name="Channel 2 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_find = waveforms_ch2[iwave_name]

    data_ave=np.average(data_to_find[0:300])
   
    data_to_find=data_to_find-data_ave

    data_min=np.min(data_to_find[:])

    idx_min=np.argmin(data_to_find[:])

    time_10percent = find_nearest(data_to_find[:], idx_min, 0.3*data_min)
    time_90percent = find_nearest(data_to_find[:], idx_min, 0.9*data_min)
          
    return time_90percent-time_10percent

def find_rising_time_ch3(filename, iwave):

    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch3 = waveforms["Channel 3"]

    total_wfm_n = len(waveforms_ch3)
    #print(total_wfm_n)
    iwave_name="Channel 3 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_find = waveforms_ch3[iwave_name]

    data_ave=np.average(data_to_find[0:300])
   
    data_to_find=data_to_find-data_ave

    data_min=np.min(data_to_find[:])

    idx_min=np.argmin(data_to_find[:])

    time_10percent = find_nearest(data_to_find[:], idx_min, 0.3*data_min)
    time_90percent = find_nearest(data_to_find[:], idx_min, 0.9*data_min)
          
    return time_90percent-time_10percent

def find_time_ch3(filename, iwave):

    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch3 = waveforms["Channel 3"]

    total_wfm_n = len(waveforms_ch3)
    #print(total_wfm_n)
    iwave_name="Channel 3 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_find = waveforms_ch3[iwave_name]

    data_ave=np.average(data_to_find[0:300])
   
    data_to_find=data_to_find-data_ave

    data_min=np.min(data_to_find[:])

    idx_min=np.argmin(data_to_find[:])

    time_10percent = find_nearest(data_to_find[:], idx_min, 0.1*data_min)
    time_50percent = find_nearest(data_to_find[:], idx_min, 0.5*data_min)

    return time_50percent, data_min

def find_time_ch2(filename, iwave):

    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch2 = waveforms["Channel 2"]
               
    total_wfm_n = len(waveforms_ch2)
    #print(total_wfm_n)
    iwave_name="Channel 2 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_find = waveforms_ch2[iwave_name]

    data_ave=np.average(data_to_find[0:300])
   
    data_to_find=data_to_find-data_ave
              
    data_ave=np.average(data_to_find[len(data_to_find)-500:len(data_to_find)])

    time_10percent = find_nearest(data_to_find, len(data_to_find), 0.1*data_ave)
    time_50percent = find_nearest(data_to_find, len(data_to_find), 0.5*data_ave)

    return time_10percent

def plot_ch2(filename, iwave):
   
    sp_time=0.0078125
    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch2 = waveforms["Channel 2"]

    total_wfm_n = len(waveforms_ch2)
    #print(total_wfm_n)
    iwave_name="Channel 2 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_plot = waveforms_ch2[iwave_name]
    
    plt.plot(sp_time*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
    plt.show()

    return 0

def plot_ch3(filename, iwave):
    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch3 = waveforms["Channel 3"]

    total_wfm_n = len(waveforms_ch3)
    print(total_wfm_n)
    iwave_name="Channel 3 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_plot = waveforms_ch3[iwave_name]
    
    #data_to_plot = waveforms_ch2["Channel 2 Seg100Data"]
    plt.plot(np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
    plt.show()

    return 0

def plot_range_ch2(filename, iwave_start, iwave_stop):

    sp_time=0.0078125
    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch2 = waveforms["Channel 2"]

    total_wfm_n = len(waveforms_ch2)
    print(total_wfm_n)
    for i in range(iwave_start,iwave_stop,1):
        iwave_name="Channel 2 Seg%dData"%(i)
        print(iwave_name)
        data_to_plot = waveforms_ch2[iwave_name]
    #plt.plot(sp_time*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
    #plt.xlim(13000,14500)
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (A.U.)")
    #plt.xlim(100,115)
    plt.show()
     
    #return 0

def plot_range_ch3(filename, iwave_start, iwave_stop):
    '''
    Open the hdf5 file and plot each waveform
    '''
    sp_time=0.0078125
    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch3 = waveforms["Channel 3"]

    total_wfm_n = len(waveforms_ch3)
    print(total_wfm_n)
    for i in range(iwave_start,iwave_stop,1):
        iwave_name="Channel 3 Seg%dData"%(i)
        print(iwave_name)
        data_to_plot = waveforms_ch3[iwave_name]  
        data_ave=np.average(data_to_plot[0:300])
        data_to_plot=data_to_plot-data_ave
        if sp_time*find_rising_time_ch3(filename,i)>6:
            plt.plot(sp_time*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (A.U.)")
    #plt.xlim(100,105)
    plt.show()


def plot_range_ch2_ch3(filename2, filename3, iwave_start, iwave_stop):

    sp_time=0.0078125
    f = h5py.File(filename2)
    waveforms = f["Waveforms"]
    waveforms_ch2 = waveforms["Channel 2"]
    f = h5py.File(filename3)
    waveforms = f["Waveforms"]
    waveforms_ch3 = waveforms["Channel 3"]

    for i in range(iwave_start,iwave_stop,1):
        iwave_name="Channel 2 Seg%dData"%(i)
        print(iwave_name)
        data_to_plot = waveforms_ch2[iwave_name]
        data_ave=np.average(data_to_plot[0:300])
        data_to_plot=data_to_plot-data_ave
        plt.plot(0.0078125*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
        #plt.plot(np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])

        iwave_name="Channel 3 Seg%dData"%(i)
        data_to_plot = waveforms_ch3[iwave_name]
        data_ave=np.average(data_to_plot[0:300])
        data_to_plot=data_to_plot-data_ave
        plt.plot(0.0078125*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
        #plt.plot(np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
    #plt.xlim(13000,14500)
    plt.show()   

def fit_gauss(x, *p0):
    A,mu,sigma=p0
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def hist_time_diff(a):
    sp_time=0.0078125

    num_bins=400
    minscale=8320*sp_time
    maxscale=12240*sp_time
    #maxscale=22240*sp_time
    
    n, bins, patches = plt.hist(a, num_bins, density=0, range=[minscale, maxscale], alpha = 1, label="Time difference between SNSPD and Laser")
    
    #plt.ylim(,) 
    #plt.xlim(0,) 
    plt.xlabel("Delta T (ns)")
    plt.ylabel("Entries")
     
    A1=np.max(n)
    #mu1=np.average(a)
    mu1=bins[np.argmax(n)]
    sigma1=0.1*mu1

    print(A1, mu1, sigma1)

    mu_max = np.argmax(n)
    delta_mu = 15
   
    bin_avg = np.average(n[10:30])
    n = n-bin_avg    

    print(bin_avg)

    print(mu_max)    
    
    try:
        fit_params,fit_cov = curve_fit(fit_gauss, bins[mu_max-delta_mu:mu_max+delta_mu], n[mu_max-delta_mu:mu_max+delta_mu], p0=(A1,mu1,sigma1))
        #fit_params,fit_cov = curve_fit(fit_gauss, bins, n, p0=(A1,mu1,sigma1))
        print(fit_params)
        A1=fit_params[0]
        mu1=fit_params[1]
        sigma1=np.abs(fit_params[2])     
    
        sigma1_error=np.sqrt(np.diag(fit_cov))[2]

        fwhm1=2.354*sigma1
     
        print("Sigma and its error")
        print(sigma1, sigma1_error)
        #plt.plot(bins[mu_max-delta_mu:mu_max+delta_mu], bin_avg+fit_gauss(bins[mu_max-delta_mu:mu_max+delta_mu],A1,mu1,sigma1), 'k--', lw=4, label="Sigma = %.2f ns"%(sigma1))
        plt.plot(bins[mu_max-delta_mu:mu_max+delta_mu], bin_avg+fit_gauss(bins[mu_max-delta_mu:mu_max+delta_mu],A1,mu1,sigma1), 'k--', lw=4, label="Sigma = %.2f+-%.2f ns"%(sigma1,sigma1_error))
    except:
        print("not able to perform fit")    

    plt.legend()
    plt.show()
    
    #plt.close()
    return 0

if (__name__=="__main__"):
 
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Draw waveforms')
    parser.add_argument('filename', help='Name of HDF5 file to process')
    parser.add_argument('channel', help='Name of channel to process')
    args = parser.parse_args()
    plot(args.filename, args.channel)
    '''
    #filename_ch2="/Users/swu1/work/spot1/timing/data/new_amp/raw/20241212_pulsetrigger_10khz_69p5tune_b1_0p21mvbias_1060nm_coincidencetrig_100ns_reset_ch3_vb0p21_newamp_waveforms_ch2.h5"
    #filename_ch3="/Users/swu1/work/spot1/timing/data/new_amp/raw/20241212_pulsetrigger_10khz_69p5tune_b1_0p21mvbias_1060nm_coincidencetrig_100ns_reset_ch3_vb0p21_newamp_waveforms_ch3.h5"
    #filename_ch2="/Users/swu1/work/spot1/timing/data/new_amp/raw/20241212_pulsetrigger_10khz_69p5tune_b1_0p21mvbias_1060nm_coincidencetrig_100ns_reset_ch3_vb0p19p5_newamp_waveforms_ch2.h5"
    #filename_ch3="/Users/swu1/work/spot1/timing/data/new_amp/raw/20241212_pulsetrigger_10khz_69p5tune_b1_0p21mvbias_1060nm_coincidencetrig_100ns_reset_ch3_vb0p19p5_newamp_waveforms_ch3.h5"
    #filename_ch2="/Users/swu1/work/spot1/data/20250106/20250106_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p24_newamp_ch2_p2.h5"
    #filename_ch3="/Users/swu1/work/spot1/data/20250106/20250106_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p24_newamp_ch3_p2.h5"
    #filename_ch2="/Users/swu1/work/spot1/data/20250122/20250122_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p19_oldamp_ch2_p1.h5"
    #filename_ch3="/Users/swu1/work/spot1/data/20250122/20250122_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p19_oldamp_ch3_p1.h5"
    filename_ch2="/Users/swu1/work/spot1/data/20250205/20250106_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p21_newamp_ch2_b1.h5"
    filename_ch3="/Users/swu1/work/spot1/data/20250205/20250106_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p21_newamp_ch3_b1.h5"
     
    #plot_range_ch2(filename_ch2, 1,2000)
    #plot_range_ch3(filename_ch3, 1,5000)

    #plot_range_ch2_ch3(filename_ch2,filename_ch3, 1,2000)

    wv_length=len(h5py.File(filename_ch2)["Waveforms"]["Channel 2"])
    
    sp_time=0.0078125
    time_diff=np.zeros(wv_length)
    time_diff_cut=np.zeros(wv_length)
    rising_time_ch3=np.zeros(wv_length)
    amp_ch3=np.zeros(wv_length)
    count=0
    for i in range(wv_length):
        time_snspd,amp_snspd=find_time_ch3(filename_ch3,i+1)      
        time_laser=find_time_ch2(filename_ch2,i+1) 
        rising_time_ch3[i]=sp_time*find_rising_time_ch3(filename_ch3,i+1)
        time_diff[i]=sp_time*(time_snspd-time_laser)
        amp_ch3[i]=amp_snspd
        print(time_snspd*sp_time,time_laser*sp_time)
        #if sp_time*find_rising_time_ch3(filename_ch3,i+1)<6 and sp_time*find_rising_time_ch3(filename_ch3,i+1)>3 and amp_ch3[i]<-16000 and amp_ch3[i]>-22000:
        #if amp_ch3[i]<-16000 and amp_ch3[i]>-22000:
        #if amp_ch3[i]<-26000 and amp_ch3[i]>-32000 and sp_time*find_rising_time_ch3(filename_ch3,i+1)<8 and sp_time*find_rising_time_ch3(filename_ch3,i+1)>2:
        if sp_time*find_rising_time_ch3(filename_ch3,i+1)<6.5:
            time_diff_cut[count]=sp_time*(time_snspd-time_laser)
            count=count+1
          
    hist_time_diff(time_diff_cut)
      
    x=rising_time_ch3
    y=time_diff    

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.figure()
    density=plt.scatter(x, y, c=z, s=50)
    plt.xlabel("SNSPD signal rising time (10-90%) (ns)")
    plt.ylabel("delta T(SNSPD-laser) (ns)")
    plt.colorbar(density)

    plt.show()
    #print(time_diff[0:2000])   
    
    #hist_time_diff(time_diff_cut)
    
    x=amp_ch3
    y=time_diff    

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.figure()
    density=plt.scatter(x, y, c=z, s=50)
    plt.xlabel("SNSPD signal amplitude (A.U.)")
    plt.ylabel("delta T(SNSPD-laser) (ns)")
    plt.colorbar(density)

    plt.show()
    
   

