'''
Code to fit rising edge to get time stamp at 50% level
Christina Wang 03/11/2025
'''

import os 
import argparse
import awkward as ak
import uproot
import numpy as np
from scipy.optimize import curve_fit

def find_nearest(a, maxid, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    try:
        idx = np.abs(a[:maxid] - a0).argmin()
    except:
        idx = 0
    #return a.flat[idx]
    return idx

# Define a linear function for fitting
def linear(x, a, b):
    return a * x + b

def linear_fit_rising_edge(signal, time, bl_length = 300, fit_range = [0.2,0.8]):
    results = {
            "rise_time": [],
            "time": [],
            "time_nofit":[],
            "amplitude": [],
            "n_fit_points":[],
            "slope":[],
            "intercept":[],
            "t_90":[],
            "t_10":[],
            'baseline': [],
            'baseline_rms':[],
            }
    for waveform_idx, waveform in enumerate(signal):
        # Perform baseline subtraction using the first 100 points
        baseline = np.mean(waveform[:bl_length])
        baseline_rms = np.sqrt(np.mean(waveform[:bl_length]**2))
        waveform_baseline_subtracted = np.abs(waveform - baseline)
        if baseline_rms> 0.1:
            # Store the results
            results["amplitude"].append(-999)
            results["time_nofit"].append(-999)
            results["time"].append(-999)
            results["rise_time"].append(-999)
            results["t_90"].append(-999)
            results["t_10"].append(-999)
            results["slope"].append(-999)
            results["intercept"].append(-999)
            results["n_fit_points"].append(-999)
            results['baseline'].append(-999)
            results['baseline_rms'].append(-999)

        else:
            # Normalize the waveform and turn to positive
            min_val = np.min(waveform_baseline_subtracted)
            max_val = np.max(waveform_baseline_subtracted)
            norm_waveform = (waveform_baseline_subtracted - min_val) / (max_val - min_val)
            # Find the first index corresponding to 10% and 90% of the waveform
            id_bl = np.where(waveform_baseline_subtracted >= baseline_rms*3)[0][0]
            idx_10 = np.where(norm_waveform[id_bl:] >= fit_range[0])[0][0] + id_bl
            idx_90 = np.where(norm_waveform[id_bl:] >= fit_range[1])[0][0] + id_bl
            # Fit the rising edge between 10% and 90%
            if idx_90-idx_10 == 0: print(f"ERROR FINDING RISING EDGE: {waveform_idx}")
            popt, _ = curve_fit(linear, time[waveform_idx][idx_10:idx_90], norm_waveform[idx_10:idx_90])
            
            # Calculate the timestamp (time at 50% of the waveform)
            timestamp = (0.5 - popt[1]) / popt[0]
            amplitude = max_val
            rise_time = time[waveform_idx][idx_90] - time[waveform_idx][idx_10]
       


            # Store the results
            results["amplitude"].append(amplitude)
            results["time_nofit"].append(time[waveform_idx][find_nearest(waveform_baseline_subtracted, np.argmax(waveform_baseline_subtracted), 0.5*max_val)])
            results["time"].append(timestamp)
            results["rise_time"].append(rise_time)
            results["t_90"].append( time[waveform_idx][idx_90])
            results["t_10"].append( time[waveform_idx][idx_10])
            results["slope"].append(popt[0])
            results["intercept"].append(popt[1])
            results["n_fit_points"].append(len(time[waveform_idx][idx_10:idx_90]))
            results['baseline'].append(baseline)
            results['baseline_rms'].append(baseline_rms)
        
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reconstruction')
    
    parser.add_argument('--channels', metavar='channels', type=str, nargs = '+', help='active channels', default = '1 2 3 4', required=True)
    parser.add_argument('--inputDir', metavar='inputDir', type=str, help='input directory',default='/raw/', required=True)
    parser.add_argument('--outputDir', metavar='outputDir', type=str, help='output directory',default='/reco/', required=True)
    parser.add_argument('--run', metavar='run', type=str, help='run number',required=True)
    args = parser.parse_args()
    active_channel = args.channels
    NUM_CHANNEL = len(active_channel)
    print("Active channels: ", active_channel)
    os.makedirs(args.outputDir, exist_ok = True)

    print(args.run)
    # Open the input ROOT file
    file_name = 'output_run'+args.run+'.root'
    input_file = uproot.open(f"{args.inputDir}/{file_name}") 
    tree = input_file["pulse"]
    print(f"output file: {args.outputDir}/{file_name}")
    output = uproot.recreate(f"{args.outputDir}/{file_name}")
    for chunk_i, input_tree in enumerate(tree.iterate(["channel", "time"], step_size=1000)):
        length = len(input_tree)
        output_data = {}
        print(f"Total number of events in chunk {chunk_i}:", length)
        for ch in range(NUM_CHANNEL):
            channel = ak.to_numpy(input_tree['channel'][:,ch,:])
            times = ak.to_numpy(input_tree['time'][:,ch,:]) * 1e9 #conver to ns from s
            results = linear_fit_rising_edge(channel, times)
            for k in results.keys():
                if k in output_data.keys():
                    output_data[k] = np.column_stack((output_data[k], results[k]))
                else: output_data[k] = results[k]
        if chunk_i == 0:
            output["pulse"] = output_data
        else: output["pulse"].extend(output_data)
    
        #if plotting: 
        #    for evt_index in range(len(channel)):
        #        plot_events(i_evt)
    #
    #def plot_events(i_evt, t_10, t_90):
    #    plt.figure(figsize=(15, 10))
    #    plt.plot(times[event_idx], clock[event_idx], label="Clock Signal")
    #    plt.title(f"Event {event_idx} Clock Signal and Fits")
    #    plt.xlabel("Time (ns)")
    #    plt.ylabel("Amplitude (mV)")
    #    rising_edge_t = times[i_evt][np.where(time[i_evt] == t_10)[0]:np.where(time[i_evt] == t_90)[0] ]
    #    plt.plot(
    #        rising_edge_t,
    #        slope * np.array(rising_edge_t) + results["biases"][event_idx][i],
    #        label=f"Fit {i}",
    #    )
    #    # Add a black vertical line at the calculated timestamp
    #    plt.axvline(x=timestamp, color="black", linestyle="--", label=f"Timestamp {i}")
    #
    #    plt.scatter(rising_edge_t, rising_edge_mV, label=f"Rising Edge {i}")
    #
    #    if len(event_timestamps) > 1:
    #        differences = [
    #            abs(event_timestamps[j] - event_timestamps[j - 1])
    #            for j in range(1, len(event_timestamps))
    #        ]
    #        clock_timestamp_differences.append(differences)
    #    else:
    #        clock_timestamp_differences.append([-10])
    #
    #    if plotting:
    #        plt.legend()
    #        plt.savefig(f"./plots/event_{event_idx}.png")
    #        plt.close()
