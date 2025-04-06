# Keysight UXR0104B
Data acquisition and reconstruction for the Keysight UXR0104B scope


## Acquisition
*  Acquire data by taking coincidence of laser trigger and SNSPD events, output files in HDF5 format
*  How to run? 

`sudo python3 acquisition.py --numEvents [nevents] --runNumber [runNumber] --laserCH [ch] --snspdCH [ch]
 --horizontalWindow [hScale] --timeoffset [hOffset] --sampleRate [sampleRate] 
--vScale1 [vScale_ch1] --vScale2 [vScale_ch2] --vScale3 [vScale_ch3] --vScale4 [vScale_ch4] 
--vPos1 [vPos_ch1] --vPos2 [vPos_ch2] --vPos3 [vPos_ch3] --vPos4 [vPos_ch4] --outputDir [outputDir]`

* To run `acquisition.py` over different bias voltages and pixels, run `acquisition_wrapper.py` 

## Reconstruction
*  The files created with the previous step are converted into ROOT TTree's using the h5py python package. This step requires the .h5 files for the channels as input.
*  First make sure you have the h5py package. If you have an existing python installation, do
`pip install h5py`
Run the conversion script
`python3 conversion_h5.py --channels 1 2 --inputDir [inputDir] --run [runNumber]`

`conversion_wrapper.py` loops over all pixels and bias voltages

### To extract the time (fit rising edge), rise time, amplitude:

`python3 reco.py --channels 1 {scope_ch[ch_i]} --inputDir {inputDir} --outputDir {outputDir} --run`

`reco_wrapper.py` loops over all pixels and bias voltages


### Final plotting

Time resolution wrt bias current are done in `plot_scripts/time_resolution.ipynb`, where EMG fit and loop over all temperature/bias current are done
Plots are saved in `plots/ADR/time_resolution_202503`
