'''
Code to record waveforms of SNSPD and laser
loop over bias voltages and SNSPD pixels
Christina Wang 03/09/2025

'''

import itertools
import sys
from time import sleep
import TimeTagger
import datetime
import pyvisa as visa
import numpy as np
import csv
import os
from TimeTagger import Coincidences, Coincidences, CoincidenceTimestamp
sys.path.append("lib")

def get_voltage(ch):
	write("CONN {},'quit'".format(ch))
	try:
		return float(query("VOLT?"))
	except:
		print("error getting voltage")
	finally:
		write("quit")
def set_voltage(ch, volt):
	write("CONN {},'quit'".format(ch))
	write('VOLT {}'.format(volt))
	write("quit")
def write(msg):sim.write(msg)
def query(msg):
	write(msg)
	return sim.read()
def turnOn(ch):
	write("CONN {}, 'quit'".format(ch))
	write('OPON')
	write('quit')
def turnOff(ch):
	write("CONN {}, 'quit'".format(ch))
	if query('EXON?'):write('OPOF')
	write('quit')

########################################################################
### define bias and readout channels, threshold, and coincidence window
########################################################################
biases = [4,1,2] #slot number of voltage source
pixels = [1,2,3] # pixel number in SNSPD
scope_ch = [2,3,4] # channel on scope
nevents = 10000
tries = 5

voltages = np.round([0.140,0.145,0.150,0.155,0.160,0.170,0.180,0.190,0.200,0.210,0.220,0.230,0.240],3)
print(voltages)

##### scope settings, assume ch1 is laser, ch2-4 are SNSPDs ####
vScale_ch1 = 0.5
vScale_ch2 = 0.05
vScale_ch3 = 0.05
vScale_ch4 = 0.05
vPos_ch1 = -1.3
vPos_ch2 = 2
vPos_ch3 = 2
vPos_ch4 = 2
hScale = 200 #ns
hOffset = 20 #ns
sampleRate = 128 #GSa/s

outputDir = r"C:/Users/Public/Documents/Infiniium/Waveforms/ADR_data/" + datetime.date.today().strftime("%Y%m%d") + "_0p2K/"
print(outputDir)
if __name__ == "__main__":
	###########################################
	# establish connection to voltage source
	##########################################
	rm=visa.ResourceManager()
	sim = rm.open_resource('ASRL/dev/ttyUSB0::INSTR')
	print("Main frame: ", query("*IDN?"))
	for ch_i, bias_ch in enumerate(biases):
		name = f"B{bias_ch}_P{pixels[ch_i]}"
		sim.write("CONN {}, 'quit'".format(bias_ch))
		print(f"Slot {bias_ch}:", query("*idn?"))
		set_voltage(bias_ch, 0.0)
		for j, v in enumerate(voltages):
        	        # set bias voltages
        	        print(v, datetime.datetime.now())
        	        set_voltage(bias_ch, voltages[j])
        	        tries_i = 0
        	        get_v = get_voltage(bias_ch)
        	        while tries_i < tries and (get_v is None or abs(voltages[j]-get_v)>=0.0001):
        	                set_voltage(bias_ch, voltages[j])
        	                get_v = get_voltage(bias_ch)
        	                print("error", tries_i, get_v, voltages[j])
        	                tries_i += 1
        	        if get_v is None or abs(voltages[j]-get_v)>=0.0001:
        	                print("channel", bias_ch, voltages[j], get_v)
        	                sys.exit("VOLTAGE IS OFF FROM TARGET")
        	        sleep(0.5)
        	        turnOn(bias_ch)
        	        # data taking with timetagger
        	        print(f"start data acquisition for BV{voltages[j]}")
        	        bv_string = str(voltages[j]).replace(".","p") 
        	        cmd = f"sudo python3 acquisition.py --numEvents {nevents} --runNumber {name}_BV{bv_string} --laserCH 1 --snspdCH {scope_ch[ch_i]}\
 --horizontalWindow {hScale} --timeoffset {hOffset} --sampleRate {sampleRate} \
--vScale1 {vScale_ch1} --vScale2 {vScale_ch2} --vScale3 {vScale_ch3} --vScale4 {vScale_ch4} \
--vPos1 {vPos_ch1} --vPos2 {vPos_ch2} --vPos3 {vPos_ch3} --vPos4 {vPos_ch4} --outputDir {outputDir}"
        	        print(cmd)
        	        os.system(cmd)
        	        turnOff(bias_ch)
