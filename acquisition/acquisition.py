
"""
VISA Control: FastFrame Acquisition
Christina Wang, 2025/03/09
For SNSPD data taking with laser
trigger is configured to coincidence trigger between laser and SNSPD channel

"""

import numpy as np
# import matplotlib.pyplot as plt
import sys
import optparse
import argparse
import signal
import os
import time
import shutil
import datetime
from shutil import copy
 
stop_asap = False

import pyvisa 

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.copytree(item, d, symlinks, ignore)
def copynew(source,destination):
    for files in source:
        shutil.copy(files,destination)

"""#################SEARCH/CONNECT#################"""
# establish communication with dpo
rm = pyvisa.ResourceManager("@py")
print(rm.list_resources())
dpo = rm.open_resource('USB0::10893::36867::MY64060141::0::INSTR')

dpo.timeout = 3000000
dpo.encoding = 'latin_1'
print(dpo.query('*idn?'))
parser = argparse.ArgumentParser(description='Run info.')

parser.add_argument('--numEvents',metavar='Events', type=str,default = 500, help='numEvents (default 500)',required=True)
parser.add_argument('--runNumber',metavar='runNumber', type=str,default = -1, help='runNumber (default -1)',required=False)
parser.add_argument('--laserCH',metavar='laserCH', type=str,default = 1, help='laser channel (default 1)',required=True)
parser.add_argument('--snspdCH',metavar='snspdCH', type=str,default = 2, help='snspd channel (default 2)',required=True)
parser.add_argument('--sampleRate',metavar='sampleRate', type=str,default = 20, help='Sampling rate (default 20)',required=True)
parser.add_argument('--horizontalWindow',metavar='horizontalWindow', type=str,default = 125, help='horizontal Window (default 125)',required=True)
parser.add_argument('--vScale1',metavar='vScale1', type=float, default= 0.02, help='Vertical scale, volts/div',required=False)
parser.add_argument('--vScale2',metavar='vScale2', type=float, default= 0.02, help='Vertical scale, volts/div',required=False)
parser.add_argument('--vScale3',metavar='vScale3', type=float, default= 0.02, help='Vertical scale, volts/div',required=False)
parser.add_argument('--vScale4',metavar='vScale4', type=float, default= 0.02, help='Vertical scale, volts/div',required=False)
parser.add_argument('--outputDir',metavar='outputDir', type=str, default= "", help='Output directory on scope',required=True)
#
parser.add_argument('--vPos1',metavar='vPos1', type=float, default= 0.02, help='Vertical offset, div',required=False)
parser.add_argument('--vPos2',metavar='vPos2', type=float, default= 0.02, help='Vertical offset, div',required=False)
parser.add_argument('--vPos3',metavar='vPos3', type=float, default= 0.02, help='Vertical offset, div',required=False)
parser.add_argument('--vPos4',metavar='vPos4', type=float, default= 0.02, help='Vertical offset, div',required=False)
parser.add_argument('--timeoffset',metavar='timeoffset', type=float, default=-130, help='Offset to compensate for trigger delay. This is the delta T between the center of the acquisition window and the trigger. (default for NimPlusX: -160 ns)',required=False)

parser.add_argument('--save',metavar='save', type=int, default= 1, help='Save waveforms',required=False)
parser.add_argument('--timeout',metavar='timeout', type=float, default= -1, help='Max run duration [s]',required=False)


args = parser.parse_args()
runNumberParam = str(args.runNumber) 
laserCH = args.laserCH
snspdCH = args.snspdCH
date = datetime.datetime.now()
savewaves = int(args.save)
timeout = float(args.timeout)
numEvents = int(args.numEvents) # number of events for each file
# numPoints = int(args.numPoints) # number of points to be acquired per event
print(savewaves)
print("timeout is ",timeout)
"""#################CONFIGURE INSTRUMENT#################"""
date = datetime.datetime.now()
hScale = float(args.horizontalWindow)*1e-9
timeoffset = float(args.timeoffset)*1e-9
samplingrate = float(args.sampleRate)*1e+9
numPoints = samplingrate*hScale

vScale_ch1 =float(args.vScale1) # in Volts for division
vScale_ch2 =float(args.vScale2) # in Volts for division
vScale_ch3 =float(args.vScale3) # in Volts for division
vScale_ch4 =float(args.vScale4) # in Volts for division

#vertical position
vPos_ch1 = float(args.vPos1)
vPos_ch2 = float(args.vPos2)
vPos_ch3 = float(args.vPos3)
vPos_ch4 = float(args.vPos4)

"""#################CONFIGURE RUN NUMBER#################"""
# increment the last runNumber by 1

if runNumberParam == -1:
	RunNumberFile = '/home/fqnet/Desktop/ScopeHandler/KeySight/Acquisition/otsdaq_runNumber.txt'
	with open(RunNumberFile) as file:
	    runNumber = int(file.read())
	print('######## Starting RUN {} ########\n'.format(runNumber))
	print('---------------------\n')
	print(date)
	print('---------------------\n')
	
	with open(RunNumberFile,'w') as f:
	    f.write(str(runNumber+1))

else: runNumber = runNumberParam

"""#################SET THE OUTPUT FOLDER#################"""
# The scope save runs localy on a shared folder with

path = args.outputDir.encode('unicode_escape').decode()

#path = r"C:\Users\Public\Documents\Infiniium\Waveforms/ADR_data/" + datetime.date.today().strftime("%Y%m%d") + "_0p2K/"
dpo.write(':DISK:MDIRectory "{}"'.format(path)) ## mkdir 
log_path = "//home/fqnet/Desktop/SNSPD_timing_KeysightScope/acquisition/Logbook.txt"

#Write in the log file
logf = open(log_path,"a+")
logf.write("\n\n#### SCOPE LOGBOOK -- RUN NUMBER {} ####\n\n".format(runNumber))
logf.write("Date:\t{}\n".format(date))
logf.write("---------------------------------------------------------\n")
logf.write("Number of events per file: {} \n".format(numEvents))
logf.write("---------------------------------------------------------\n\n")


"""#################SCOPE HORIZONTAL SETUP#################"""
# dpo setup

dpo.write(':STOP;*OPC?')

dpo.write(':TIMebase:RANGe {}'.format(hScale)) ## Sets the full-scale horizontal time in s. Range value is ten times the time-per division value.
dpo.write(':TIMebase:REFerence:PERCent 50') ## percent of screen location
dpo.write(':ACQuire:SRATe:ANALog {}'.format(samplingrate))
dpo.write(':TIMebase:POSition {}'.format(timeoffset)) ## offset
dpo.write(':ACQuire:MODE SEGMented') ## fast frame/segmented acquisition mode
dpo.write(':ACQuire:SEGMented:COUNt {}'.format(numEvents)) ##number of segments to acquire
#dpo.write(':ACQuire:POINts:ANALog {}'.format(numPoints))
dpo.write(':ACQuire:INTerpolate 0') ## interpolation is set off (otherwise its set to auto, which cause errors downstream)

"""#################SCOPE CHANNELS BANDWIDTH#################"""
# dpo.write(':ACQuire:BANDwidth MAX') ## set the bandwidth to maximum
# dpo.write('CHANnel1:ISIM:BANDwidth 2.0E+09')
# dpo.write('CHANnel2:ISIM:BANDwidth 2.0E+09')
# dpo.write('CHANnel3:ISIM:BANDwidth 2.0E+09')
# dpo.write('CHANnel4:ISIM:BANDwidth 2.0E+09')

# dpo.write('CHANnel1:ISIM:BWLimit 1')
# dpo.write('CHANnel2:ISIM:BWLimit 1')
# dpo.write('CHANnel3:ISIM:BWLimit 1')
# dpo.write('CHANnel4:ISIM:BWLimit 1')

#dpo.write(':ACQuire:BANDwidth 2.E9')
"""#################SCOPE VERTICAL SETUP#################"""
#vScale expressed in Volts
dpo.write('CHANnel1:SCALe {}'.format(vScale_ch1))
dpo.write('CHANnel2:SCALe {}'.format(vScale_ch2))
dpo.write('CHANnel3:SCALe {}'.format(vScale_ch3))
dpo.write('CHANnel4:SCALe {}'.format(vScale_ch4))

dpo.write('CHANnel1:OFFSet {}'.format(-vScale_ch1 * vPos_ch1))
dpo.write('CHANnel2:OFFSet {}'.format(-vScale_ch2 * vPos_ch2))
dpo.write('CHANnel3:OFFSet {}'.format(-vScale_ch3 * vPos_ch3))
dpo.write('CHANnel4:OFFSet {}'.format(-vScale_ch4 * vPos_ch4))


logf.write("VERTICAL SETUP\n")
logf.write('- CH1: vertical scale set to {} V for division\n'.format(vScale_ch1))
logf.write('- CH2: vertical scale set to {} V for division\n'.format(vScale_ch2))
logf.write('- CH3: vertical scale set to {} V for division\n'.format(vScale_ch3))
logf.write('- CH4: vertical scale set to {} V for division\n\n'.format(vScale_ch4))


"""#################TRIGGER SETUP#################"""
dpo.write("TRIGger:MODE SEQuence")
dpo.write(f"TRIGger:LEVel CHANnel{laserCH},0.5")
dpo.write(f"TRIGger:LEVel CHANnel{snspdCH},-0.06")
dpo.write(":TRIGger:SEQuence:TERM1 EDGE1")
dpo.write(":TRIGger:SEQuence:TERM2 EDGE2")
dpo.write("TRIGger:SEQuence:RESet:ENABle 1")
dpo.write("TRIGger:SEQuence:RESet:TYPE TIME")
dpo.write("TRIGger:SEQuence:RESet:TIME 1e-7")
dpo.write(f"TRIGger:EDGE1:SOURce CHAN{laserCH}")
dpo.write("TRIGger:EDGE1:SLOPe POSitive")
dpo.write(f"TRIGger:EDGE2:SOURce CHAN{snspdCH}")
dpo.write("TRIGger:EDGE2:SLOPe NEGative")
print('Horizontal, vertical, and trigger settings configured.\n')

status = ""
status = "busy"


"""#################DATA TRANSFERRING#################"""
# configure data transfer settings
time.sleep(2)

print(dpo.write(':CDISplay'))

dpo.write('*CLS;:SINGle')
start = time.time()
end_early = False
while True:
	if (int(dpo.query(':ADER?')) == 1): 
		print("Acquisition complete")
		break
	else:
		#print "Still waiting" 
		time.sleep(0.1)
		if not savewaves and timeout > 0 and time.time() - start > timeout:
			end_early=True
			dpo.write(':STOP;*OPC?')
			break

end = time.time()
#print(dpo.query('*OPC?'))
# print("Trigger!")


duration = end - start
trigRate = float(numEvents)/duration
if not end_early: print("\nRun duration: %0.2f s. Trigger rate: %.2f Hz\n" % (duration,trigRate))
else: print("\nRun duration: %0.2f s. Trigger rate: unknown\n" % (duration) )
if savewaves: 
	dpo.write(':DISK:SEGMented ALL') ##save all segments (as opposed to just the current segment)
	print(dpo.query('*OPC?'))
	print("Ready to save all segments")
	time.sleep(0.5)
	for i in [int(laserCH),int(snspdCH)]:
		dpo.write(f':DISK:SAVE:WAVeform CHANnel{i} ,"{path}\\Wavenewscope_CH{i}_{runNumber}",H5,ON')

		print(dpo.query('*OPC?'))
		print(f"Saved Channel {i} waveform")
		time.sleep(1)
else: print("Skipping saving step.")


dpo.close()


