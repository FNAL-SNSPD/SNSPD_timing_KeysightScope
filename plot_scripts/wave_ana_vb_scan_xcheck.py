#!/usr/bin/env python2.7

import math
import h5py
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.stats import gaussian_kde
from lmfit.models import GaussianModel

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
    waveforms_ch2 = waveforms["Channel 1"]

    total_wfm_n = len(waveforms_ch2)
    #print(total_wfm_n)
    iwave_name="Channel 1 Seg%dData"%(iwave)
    #print(iwave_name)
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
    waveforms_ch3 = waveforms["Channel 2"]

    total_wfm_n = len(waveforms_ch3)
    #print(total_wfm_n)
    iwave_name="Channel 2 Seg%dData"%(iwave)
    #print(iwave_name)
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
    #print(iwave_name)
    data_to_find = waveforms_ch3[iwave_name]

    data_ave=np.average(data_to_find[0:300])
   
    data_to_find=data_to_find-data_ave

    data_min=np.min(data_to_find[:])

    idx_min=np.argmin(data_to_find[:])

    time_10percent = find_nearest(data_to_find[:], idx_min, 0.1*data_min)
    time_50percent = find_nearest(data_to_find[:], idx_min, 0.5*data_min)

    return time_50percent, data_min

def find_time_ch3_fit(filename, iwave):

    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch3 = waveforms["Channel 2"]

    total_wfm_n = len(waveforms_ch3)
    #print(total_wfm_n)
    iwave_name="Channel 2 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_find = waveforms_ch3[iwave_name]

    data_ave=np.average(data_to_find[0:300])

    data_to_find=data_to_find-data_ave

    data_min=np.min(data_to_find[:])

    idx_min=np.argmin(data_to_find[:])

    time_20percent = find_nearest(data_to_find[:], idx_min, 0.2*data_min)
    time_80percent = find_nearest(data_to_find[:], idx_min, 0.8*data_min)

    time_bin = np.arange(time_20percent,time_80percent,1)
    snspd_wfm = data_to_find[time_20percent:time_80percent]

    A1=-100
    B1=1000

    try:
        fit_params,fit_cov = curve_fit(fit_linear, time_bin, snspd_wfm, p0=(A1,B1))
        #fit_params,fit_cov = curve_fit(fit_gauss, bins, n, p0=(A1,mu1,sigma1))
        #print(fit_params)
        A1=fit_params[0]
        B1=fit_params[1]
    except:
        print("not able to perform linear fit to SNSPD wfm")
   
    '''
    plt.plot(np.arange(0,len(data_to_find[:]),1),data_to_find[:])
    plt.plot(time_bin,fit_linear(time_bin,A1,B1),color='red')
    plt.show()
    '''

    #print (0.1*(0.5*data_min-B1)/A1)
    return 1.0*(0.5*data_min-B1)/A1, data_min

def find_time_ch2_fit(filename, iwave):

    f = h5py.File(filename)
    waveforms = f["Waveforms"]
    waveforms_ch2 = waveforms["Channel 1"]

    total_wfm_n = len(waveforms_ch2)
    #print(total_wfm_n) 
    iwave_name="Channel 1 Seg%dData"%(iwave)
    print(iwave_name)
    data_to_find = waveforms_ch2[iwave_name]

    data_ave=np.average(data_to_find[0:300])

    data_to_find=data_to_find-data_ave

    data_ave=np.average(data_to_find[len(data_to_find)-500:len(data_to_find)])

    time_20percent = find_nearest(data_to_find, len(data_to_find), 0.2*data_ave)
    time_80percent = find_nearest(data_to_find, len(data_to_find), 0.8*data_ave)    

    time_bin = np.arange(time_20percent,time_80percent,1)
    laser_wfm = data_to_find[time_20percent:time_80percent]
       
    A1=100
    B1=1000

    try:
        fit_params,fit_cov = curve_fit(fit_linear, time_bin, laser_wfm, p0=(A1,B1))
        #fit_params,fit_cov = curve_fit(fit_gauss, bins, n, p0=(A1,mu1,sigma1))
        #print(fit_params)
        A1=fit_params[0]
        B1=fit_params[1]
    except:
        print("not able to perform linear fit to laser wfm")
    
    '''
    plt.plot(np.arange(0,len(data_to_find[:]),1),data_to_find[:])
    plt.plot(time_bin,fit_linear(time_bin,A1,B1),color='red')
    plt.show()
    '''

    #print(0.1*(0.5*data_ave-B1)/A1)
    return 1.0*(0.5*data_ave-B1)/A1

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
    waveforms_ch2 = waveforms["Channel 1"]

    total_wfm_n = len(waveforms_ch2)
    print(total_wfm_n)
    for i in range(iwave_start,iwave_stop,1):
        iwave_name="Channel 1 Seg%dData"%(i)
        print(iwave_name)
        data_to_plot = waveforms_ch2[iwave_name]
    plt.plot(sp_time*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
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
    waveforms_ch3 = waveforms["Channel 2"]

    total_wfm_n = len(waveforms_ch3)
    print(total_wfm_n)
    for i in range(iwave_start,iwave_stop,1):
        iwave_name="Channel 2 Seg%dData"%(i)
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
    waveforms_ch2 = waveforms["Channel 1"]
    f = h5py.File(filename3)
    waveforms = f["Waveforms"]
    waveforms_ch3 = waveforms["Channel 2"]

    for i in range(iwave_start,iwave_stop,1):
        iwave_name="Channel 1 Seg%dData"%(i)
        print(iwave_name)
        data_to_plot = waveforms_ch2[iwave_name]
        data_ave=np.average(data_to_plot[0:300])
        data_to_plot=data_to_plot-data_ave
        plt.plot(0.0078125*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
        #plt.plot(np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])

        iwave_name="Channel 2 Seg%dData"%(i)
        data_to_plot = waveforms_ch3[iwave_name]
        data_ave=np.average(data_to_plot[0:300])
        data_to_plot=data_to_plot-data_ave
        plt.plot(0.0078125*np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
        #plt.plot(np.arange(0,len(data_to_plot[:]),1),data_to_plot[:])
    #plt.xlim(13000,14500)
    plt.show()   

def fit_gauss(x, p0, p1, p2):
    A=p0
    mu=p1
    sigma=p2
    #,mu,sigma=p0
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def fit_linear(x, *p0):
    A,B=p0
    return A*x+B
     
def fit_emg(x, p0, p1, p2,p3):
    #amp,mu,sigma,lambd = p0
    #amp=p0
    #mu=p1
    #sigma=p2
    #lambd=p3
    #mu = p0[1]
    #sigma = p0[2]
    #lambd= p0[3]   
    erfc_part = math.erfc((p1+p3*p2**2-x)/(p2*2**0.5))
    return p1*p3/2*math.exp(p3/2 *(2*p1 + p3*p2**2-2*x)) * erfc_part

def hist_time_diff(a):
    sp_time=0.0078125

    num_bins=300
    #minscale=8320*sp_time
    minscale=4000*sp_time
    maxscale=12240*sp_time
    #maxscale=22240*sp_time
    
    n, bins, patches = plt.hist(a, num_bins, density=0, range=[minscale, maxscale], alpha = 1, label="Time difference between SNSPD and Laser")
    
    #plt.ylim(,) 
    #plt.xlim(0,) 
    plt.xlabel("Delta T (ns)")
    plt.ylabel("Entries")
     
    A1=np.max(n)
    mu1=bins[np.argmax(n)]
    sigma1=0.1*mu1
    #lambda1=0.1*mu1
    
    print("initial guess:")
    print(A1, mu1, sigma1)

    mu_max = np.argmax(n)
    delta_mu = 30
    
    #bin_avg = np.average(n[1:3])
    #n = n-bin_avg    
    
    model = GaussianModel()
    params = model.make_params(center=mu1, amplitude=A1, sigma=sigma1)  
    result = model.fit(n[mu_max-delta_mu:mu_max+delta_mu], params, x=bins[mu_max-delta_mu:mu_max+delta_mu])
    print(result.redchi)     
    print(result.nfree)     
    print(result.params['center'].value)     
    print(result.params['amplitude'].value)     
    print(result.params['sigma'].value)     
    
    A1=result.params['amplitude'].value
    mu1=result.params['center'].value
    sigma1=result.params['sigma'].value
          
    plt.plot(bins[mu_max-delta_mu:mu_max+delta_mu], fit_gauss(bins[mu_max-delta_mu:mu_max+delta_mu],A1,mu1,sigma1), 'k--', lw=1, label="Sigma = %.2f ns, chi2/ndf = %.2f/%.2f"%(sigma1,result.redchi,result.nfree))

    '''
    try:
        fit_params,fit_cov = curve_fit(fit_gauss, bins[mu_max-delta_mu:mu_max+delta_mu+1], n[mu_max-delta_mu:mu_max+delta_mu+1], p0=(A1,mu1,sigma1))
        #fit_params,fit_cov = curve_fit(fit_emg, bins[mu_max-delta_mu:mu_max+delta_mu+1], n[mu_max-delta_mu:mu_max+delta_mu+1], p0=(A1,mu1,sigma1,lambda1))
        print(fit_params)
        A1=fit_params[0]
        mu1=fit_params[1]
        sigma1=np.abs(fit_params[2])     
        #lambda1=np.abs(fit_params[3])     
    
        sigma1_error=np.sqrt(np.diag(fit_cov))[2]

        print("Sigma and its error")
        print(sigma1, sigma1_error)
        plt.plot(bins[mu_max-delta_mu:mu_max+delta_mu], fit_gauss(bins[mu_max-delta_mu:mu_max+delta_mu],A1,mu1,sigma1), 'k--', lw=4, label="Sigma = %.2f ns"%(sigma1))
        #plt.plot(bins[mu_max-delta_mu:mu_max+delta_mu], fit_emg(bins[mu_max-delta_mu:mu_max+delta_mu],A1,mu1,sigma1,lambda1), 'k--', lw=1, label="Sigma = %.2f+-%.2f ns"%(sigma1,sigma1_error))
    except:      
        print("not able to perform fit")    
    '''

    #plt.plot(bins[mu_max-delta_mu:mu_max+delta_mu+1], bin_avg+fit_emg(bins[mu_max-delta_mu:mu_max+delta_mu+1],A1,mu1,sigma1,lambda1), 'k--', lw=1, label="Sigma = %.2f+ ns"%(sigma1))
    #plt.plot(bins[mu_max-delta_mu:mu_max+delta_mu+1], bin_avg+fit_emg(bins[mu_max-delta_mu:mu_max+delta_mu+1],A1,mu1,sigma1,lambda1), 'k--', lw=1, label="Sigma = %.2f+ ns"%(sigma1))
    #plt.xlim([74,81])
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
    #filename_ch2="/Users/swu1/work/spot1/data/20250205/20250106_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p21_newamp_ch2_b1.h5"
    #filename_ch3="/Users/swu1/work/spot1/data/20250205/20250106_pulsetrigger_10khz_50p0tune_1060nm_coincidencetrig_100ns_reset_vb0p21_newamp_ch3_b1.h5"
     
    filename_ch2="/Users/swu1/Downloads/Wavenewscope_CH1_B4_P1_BV0p2.h5" #//CH1 laser
    filename_ch3="/Users/swu1/Downloads/Wavenewscope_CH2_B4_P1_BV0p2.h5" #//CH2 snspd
    
    print(len(h5py.File(filename_ch2)["Waveforms"]["Channel 1"]))
    print(len(h5py.File(filename_ch3)["Waveforms"]["Channel 2"]))
                                   
    #plot_range_ch2(filename_ch2, 1,2000)
    #plot_range_ch3(filename_ch3, 1,2000)

    #plot_range_ch2_ch3(filename_ch2,filename_ch3, 1,2000)

    wv_length=len(h5py.File(filename_ch2)["Waveforms"]["Channel 1"])
    
    sp_time=0.0078125
    time_diff=np.zeros(wv_length)
    time_diff_cut=np.zeros(wv_length)
    rising_time_ch3=np.zeros(wv_length)
    amp_ch3=np.zeros(wv_length)
    count=0
    for i in range(wv_length):
        time_snspd,amp_snspd=find_time_ch3_fit(filename_ch3,i+1)      
        time_laser=find_time_ch2_fit(filename_ch2,i+1) 
        rising_time_ch3[i]=sp_time*find_rising_time_ch3(filename_ch3,i+1)
        time_diff[i]=sp_time*(time_snspd-time_laser)
        amp_ch3[i]=amp_snspd
        print(time_snspd*sp_time,time_laser*sp_time)
        #if sp_time*find_rising_time_ch3(filename_ch3,i+1)<6 and sp_time*find_rising_time_ch3(filename_ch3,i+1)>3 and amp_ch3[i]<-16000 and amp_ch3[i]>-22000:
        #if amp_ch3[i]<-16000 and amp_ch3[i]>-22000:
        #if amp_ch3[i]<-26000 and amp_ch3[i]>-32000 and sp_time*find_rising_time_ch3(filename_ch3,i+1)<8 and sp_time*find_rising_time_ch3(filename_ch3,i+1)>2:
        if sp_time*find_rising_time_ch3(filename_ch3,i+1)<660:
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
    plt.ylim([74,82])
    plt.xlim([3,11])
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
    
   

