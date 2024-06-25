# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:08:33 2024

@author: leschapasse
"""


import csv
import numpy as np
from scipy import signal
import pylsl
import explorepy
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import butter
from scipy.signal import lfilter

  
def explorepy_script(start_time, duration, current_time, patient_name,batch_size):
    
   # Initialize ExplorePy device
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_8447")
    explorer.set_sampling_rate(sampling_rate=250)
    explorer.set_channels(channel_mask="00011111") #five channels
    explorer.push2lsl(duration)
    
    # Initialize buffers for each channel
    buffer_channel1 = np.empty(1000)
    buffer_channel2 = np.empty(1000)
    buffer_channel3 = np.empty(1000)
    buffer_channel4 = np.empty(1000)
    #buffer_timestamp = np.empty(1000)
    derivative_result1 = np.zeros(1000)
    derivative_result2 = np.zeros(1000)
    sub_result1 = np.zeros(1000)
    sub_result2 = np.zeros(1000)
    buffer_idx = 0
    buffer_update_freq = 50  # Frequency to update buffer
    
    # Bandpass filter parameters
    lowcut = 0.1
    highcut = 20.0
    fs = 250.0  # Sampling rate
    nyquist = 0.5 * fs
    b, a = butter(2, [lowcut / nyquist, highcut / nyquist], btype="band")

    # Moving average filter parameters
    kernel_size = 36
    moving_avg_kernel = np.ones(kernel_size) / kernel_size
    amplification_factor = 2.0

    # LSL stream setup
    streams = pylsl.resolve_stream('name', 'Explore_8447_ExG') #Mentallab
    #streams = pylsl.resolve_stream('name', 'EOGData')
    inlet = pylsl.StreamInlet(streams[0])

    # Time tracking and storage lists
    derivative_result1_list = []
    derivative_result2_list = []
    sub_result1_list = []
    sub_result2_list = []

    prev_filtered_result1 = 0  # Adjust size based on expected output after convolution
    prev_filtered_result2 = 0
   # prev_time = 0#maybe uncomment ?

    end_time = start_time + duration
    first=None
    batch_data = []
    
    while time.time() < end_time:
        sample, timestamp = inlet.pull_sample()
        if first == None :
            start_elapsed=time.time()
            first=1
        buffer_channel1[buffer_idx] = sample[0] #channel number - 1
        buffer_channel2[buffer_idx] = sample[1]
        buffer_channel3[buffer_idx] = sample[2]
        buffer_channel4[buffer_idx] = sample[4]
        
        buffer_idx += 1
 
        if buffer_idx == len(buffer_channel1):
            #start_elapsed = time.time()
            buffer_idx = 999 - buffer_update_freq # peut etre change pour real time

            # Apply bandpass filter to the data
            filtered_result1 = lfilter(b, a, buffer_channel1 - buffer_channel2)#vertical
            filtered_result2 = lfilter(b, a, buffer_channel3 - buffer_channel4)

            # Convolution operation
            filtered_result1 = np.convolve(filtered_result1, moving_avg_kernel, mode='valid')
            filtered_result2 = np.convolve(filtered_result2, moving_avg_kernel, mode='valid')

            # Calculate derivatives
            current_time1=timestamp
            time_diff = current_time1
            
            derivative1 = (filtered_result1 - prev_filtered_result1) / time_diff
            derivative2 = (filtered_result2 - prev_filtered_result2) / time_diff
            
            derivative1 = derivative1 * amplification_factor
            derivative2 = derivative2 * amplification_factor
            # Update previous values
            prev_filtered_result1 = filtered_result1
            prev_filtered_result2 = filtered_result2
                        
            # Calculer le temps écoulé depuis le début de la session pour cette valeur
            derivative_result1=np.append(derivative_result1_list, derivative1)
            derivative_result2=np.append(derivative_result2_list, derivative2)

            sub_result1=np.append(sub_result1_list, filtered_result1)
            sub_result2=np.append(sub_result2_list, filtered_result2)
            
            derivative_result1 = derivative_result1[-1000:]
            derivative_result2 = derivative_result2[-1000:]
            
            sub_result1 = sub_result1[-1000:]
            sub_result2 = sub_result2[-1000:]
            
            # derivative_result1_list.extend(derivative_result1[-buffer_update_freq:])
            # derivative_result2_list.extend(derivative_result2[-buffer_update_freq:])
            
            # sub_result1_list.extend(sub_result1[-buffer_update_freq:])
            # sub_result2_list.extend(sub_result2[-buffer_update_freq:])
            
            batch_data.append([derivative1, derivative2, sub_result1, sub_result2])       

            # Shift the buffer for the next iteration
            buffer_channel1[:buffer_idx + 1] = buffer_channel1[buffer_update_freq:]
            buffer_channel2[:buffer_idx + 1] = buffer_channel2[buffer_update_freq:]
            buffer_channel3[:buffer_idx + 1] = buffer_channel3[buffer_update_freq:]
            buffer_channel4[:buffer_idx + 1] = buffer_channel4[buffer_update_freq:]
                
        if len(batch_data)>= batch_size:
            # print(batch_data)
            # print(len(batch_data))
            print(np.shape(batch_data))
            batch_data = np.array(batch_data)  # Convert list to numpy array
            #batch_data = np.array(batch_data).reshape((batch_size,1,4))
            #model_predictions = online_model(model, batch_data)
            batch_data = []

    duration_elapsed_list = end_time - start_elapsed
    start_elapsed_time = np.arange(0, duration_elapsed_list, 0.004)
    time_diff_elapsed = start_elapsed - start_time
    start_elapsed_time += time_diff_elapsed
    
    #Save data to a CSV file in Downloads
    with open(f"{os.path.expanduser('~')}/Downloads/Recording20s", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(["Derivative 1", "Derivative 2", "Sub Result 1", "Sub Result 2"])
        # Write data rows
        for i in range(len(derivative1)):
            row = [derivative1[i], derivative2[i], sub_result1[i], sub_result2[i]]
            writer.writerow(row)
            
    print("Data saved successfully")

    return batch_data
    
    explorer.disconnect()
    
    
 #%% MAIN   
def main():
    # Set the duration for the data collection process
    duration = 20
    
    # Collecting user input for patient information and electrode configuration
    patient_name = input("Please enter the patient's name: ")

    # Record the start time and current time
    start_time = time.time()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

                     
    record_interval = 1 / 250  # Record data every 4 milliseconds
    current_time1 = time.time()
    batch_size=32
    last_record_time = 0
    elapsed_time = current_time1 - start_time
   
    batch_data = explorepy_script(start_time, duration, current_time, patient_name, batch_size)

    ## Record data at specified intervals  
    try:
        if elapsed_time - last_record_time >= record_interval:   
           last_record_time = elapsed_time

    except FileNotFoundError as e:
        print("Error reading the file:", e)
    except Exception as e:
        print("An error occurred while processing the data:", e)

    # print("derivative_result1_list size:", np.shape(derivative_result1_list))
    # print("derivative_result2_list size:", np.shape(derivative_result2_list))
    # print("sub_result1_list size:", np.shape(sub_result1_list))
    # print("sub_result2_list size:", np.shape(sub_result2_list))

    
if __name__ == "__main__":
    main()


    