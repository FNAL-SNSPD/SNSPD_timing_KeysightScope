import numpy as np
import subprocess
import os
########################################################################
### define bias and readout channels, threshold, and coincidence window
########################################################################
biases = [4,1,2] #slot number of voltage source
pixels = [1,2,3] # pixel number in SNSPD
scope_ch = [2,3,4] # channel on scope

voltages = np.round([ 0.130,0.135,0.140,0.145,0.150,0.155,0.160,0.170,0.180,0.190,0.200,0.210,0.220, 0.230,0.240],3)
print(voltages)
inputDirBase = "/eos/uscms/store/user/christiw/SNSPD_data/ADR_time_resolution_202503/"

tempToDir = {
0.2: "20250311_0p2K",
0.5: "20250310_0p5K",
0.8: "20250309_0p8K",
1:"20250310_1p0K",

 }
if __name__ == "__main__":
    for k, v in tempToDir.items():
        inputDir = f"{inputDirBase}/{v}/raw/"
        outputDir = inputDir.replace("raw","reco")
        for ch_i, bias_ch in enumerate(biases):
            name = f"B{bias_ch}_P{pixels[ch_i]}"
            processes = []
            for j, v in enumerate(voltages):
                bv_string = str(voltages[j]).replace(".","p")
                if not os.path.exists(inputDir+f'output_run{name}_BV{bv_string}.root'):
                    print("FILE NOT FOUND", f"{name}_BV{bv_string}")
                    continue
                cmd = f"python3 reco.py --channels 1 {scope_ch[ch_i]} --inputDir {inputDir} --outputDir {outputDir} --run {name}_BV{bv_string}"
                print(cmd)
                os.system(cmd)
                #processes.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            #exit_codes = [p.wait() for p in processes]
            #print(exit_codes)
