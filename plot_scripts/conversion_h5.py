import ROOT
import h5py
import sys
import optparse
import argparse
import time
import numpy as np
import os,sys
import ctypes
from array import array

parser = argparse.ArgumentParser(description='Reconstruction')

parser.add_argument('--channels', metavar='channels', type=str, nargs = '+', help='active channels', default = '1 2 3 4', required=True)
parser.add_argument('--inputDir', metavar='inputDir', type=str, help='input directory',default='/20210308/', required=True)
parser.add_argument('--run', metavar='run', type=str, help='run number',required=True)
args = parser.parse_args()
print(args.channels)
active_channel = args.channels
NUM_CHANNEL = len(active_channel)
print("Active channels: ", active_channel)
##---Read the input files
f = []
Channel = []
for i in active_channel:
    f.append(h5py.File(args.inputDir+'Wavenewscope_CH'+str(i)+'_'+args.run+'.h5', 'r')) #---Channel 1
    Channel.append(f[-1]['Waveforms']['Channel '+str(i)])
##---Prepare output file
outputFile = args.inputDir+'output_run'+args.run+'.root'
#if os.path.exists(outputFile):sys.exit("OUTPUT ALREADY EXISTS!!! " + outputFile)  
outRoot = ROOT.TFile(outputFile, "RECREATE")
outTree = ROOT.TTree("pulse","pulse")

length = f[0]['Waveforms']['Channel '+str(active_channel[0])].attrs['NumPoints']
print(length)
i_evt = np.zeros(1,dtype=np.dtype("u4"))
runNum = np.zeros(1,dtype=str)
channel = np.zeros([NUM_CHANNEL,length],dtype=np.float32)
time = np.zeros([NUM_CHANNEL,length],dtype=np.float32)

outTree.Branch('i_evt',i_evt,'i_evt/i')
outTree.Branch('runNum',runNum,'runNum/i')
outTree.Branch( 'channel', channel, 'channel['+str(NUM_CHANNEL)+']['+str(length)+']/F' )
outTree.Branch( 'time', time, 'time['+str(NUM_CHANNEL)+']['+str(length)+']/F')

#---Begin reconstruction method

n_events = f[0]['Waveforms']['Channel '+str(active_channel[0])].attrs['NumSegments'] #---number of events or segments
n_points = f[0]['Waveforms']['Channel '+str(active_channel[0])].attrs['NumPoints']   #---number of points acquired for each segment (same for each channel)

#---Store the time(same for each channel and every event)
time_temp = []
for point in range(n_points):
    time_temp.append(Channel[0].attrs['XDispOrigin'] + point*Channel[0].attrs['XInc'])
for i in range(NUM_CHANNEL):
    time[i] = time_temp

for event in range(n_events):
    i_evt[0] = event
    runNum[0] = args.run
    for i,ch in enumerate(active_channel):
        channel[i] = Channel[i]['Channel '+str(ch)+' Seg'+str(event+1)+'Data'][()]
    outTree.Fill()

outRoot.cd()
outTree.Write()
outRoot.Close()
print("outputFile: ", outputFile)
