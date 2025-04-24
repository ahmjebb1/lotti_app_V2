### IMPORTS ###

import os
import time
import tkinter as tk
from tkinter import Frame, StringVar, Label, Entry, Button, Toplevel, filedialog, messagebox, ttk, LabelFrame
import pandas as pd
import numpy as np
import librosa
import noisereduce as nr
import pickle
from scipy.signal import find_peaks
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import re

### DIRECTORIES ###

#Find out working directory
print("Current working directory:", os.getcwd())

#set folder to save outputs
output_folder = os.getcwd()

### LOADING ###

#List of files testes

#240508_Nest_7_J_R1.wav, offset = 17.35, duration = 0.1
#240315_Nest_4_R1.wav, offset = 213.3682, duration = 0.13
#240404_Nest_47_R4, , offset = 205.3433, duration = 0.13

#Testing librosa.load
    #Testing with a random section of a random file
original_y, sr = librosa.load("C:/Users/bi1ahj/Desktop/Selection_Tables/250210_ML50_churr_dataset/Dataset/2024/FBL/Audio Data/240322_Nest_35_R1.wav", offset = 65.8848, duration = 0.13, sr = 48000)

# Compute the Mel spectrogram for the denoised signal (Not originally in lotti App)
original_S = librosa.feature.melspectrogram(y = original_y, sr=sr, n_fft=768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

#Summarise the shape
print(f"ORIGINAL LOADING:")
print(f"Sample rate: {sr}")
print(f"Waveform duration (calculated): {len(original_y) / sr:.3f} seconds")
print(f"First 10 values of y: {original_y[:10]}")
print(f"Min: {original_y.min():.4f}, Max: {original_y.max():.4f}")

#Summarise the spectrogram
print(f"DENOISING STEP:")
print(f"Denoised Waveform Min: {original_S.min()}, Max: {original_S.max()}\n")
print(f"Top 5 values: {original_S[:5]}\n")
print(f"Bottom 5 values: {original_S[-5:]}\n")

#Plot the original shape as a mel spectrogram
#plt.figure(figsize=(10, 6))
librosa.display.specshow(original_S, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar()
plt.title("Original Mel Spectrogram before any data transformations")
plt.tight_layout()

# Save the spectrogram to the working directory
plt.savefig(os.path.join(output_folder, "pl1_mel_spectrogram_original.png"))
plt.close()

### NOISE REDUCTION ###

#Run the noise reduction
y = nr.reduce_noise(y=original_y, sr=sr, thresh_n_mult_nonstationary=12, n_fft=768)

# Compute the Mel spectrogram for the denoised signal
S = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

#Summarise
print(f"DENOISING STEP:")
print(f"Denoised Waveform Min: {y.min()}, Max: {y.max()}\n")
print(f"Top 5 values: {y[:5]}\n")
print(f"Bottom 5 values: {y[-5:]}\n")

#Plot the denoised signal as a mel spectrogram
#plt.figure(figsize=(10, 6))
librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar()
plt.title("Mel Spectrogram after denoising")
plt.tight_layout()

# Save the spectrogram to the working directory
plt.savefig(os.path.join(output_folder, "pl2_mel_spectrogram_denoised.png"))
plt.close()

### DB SPEC ###
S_dB = librosa.amplitude_to_db(S, ref=np.max)

#Summarize
print(f"MAGNITUDE SPEC:")
print(f"DB Spec Min: {S_dB.min()}, Max: {S_dB.max()}\n")
print(f"Top 5 values: {S_dB[:5]}\n")
print(f"Bottom 5 values: {S_dB[-5:]}\n")

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram After Noise Reduction And Converted to dB")
plt.tight_layout()

# Save the spectrogram to the working directory
plt.savefig(os.path.join(output_folder, "pl3_mel_spectrogram_dB.png"))
plt.close()

### NORMALIZE ###

# Normalize the array to the range [0, 1]
#S_dB_norm = librosa.util.normalize(S_dB, axis=0)
# Manually scale to [0, 1]
S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

#Summarise
# Inspecting shape and basic statistics
print(f"NORMALIZED DATA:")
print(f"Shape of S_dB_norm: {S_dB_norm.shape}")
print(f"Min value: {S_dB_norm.min()}")
print(f"Max value: {S_dB_norm.max()}")
print(f"Mean value: {S_dB_norm.mean()}")
print(f"Standard deviation: {S_dB_norm.std()}")

### PLOT ###

# Save the plot to the current working directory
plt.figure(figsize=(10, 6))
plt.imshow(S_dB_norm, aspect='auto', origin='lower', cmap='viridis')
plt.title("Normalized Mel Spectrogram")
plt.xlabel("Time (frames)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "pl4_normalized_mel_spectrogram.png"))
plt.close()

### NEW SPEC OF DOM FREQ ###

# Find the index of the frequency with the highest magnitude at each time point
dominant_indices = np.argmax(S, axis=0)

# Create a new spectrogram with only the dominant frequencies set to their original amplitude
S_dominant = np.zeros(S.shape)

# Create a 2D array of indices for proper indexing
time_indices = np.arange(S.shape[1])
indices_array = np.stack((dominant_indices, time_indices), axis=-1)
S_dominant[tuple(indices_array.T)] = S[tuple(indices_array.T)]

#Summarize

# Inspect the array to see if there are regions of 0s causing the border
print(f"Min value of S_dominant: {S_dominant.min()}")
print(f"Max value of S_dominant: {S_dominant.max()}")

#Plot
# Set up plot
plt.figure(figsize=(10, 6))
# Plot dominant-frequency-only spectrogram
librosa.display.specshow(librosa.amplitude_to_db(S_dominant + 1e-10, ref=np.max),  # add small value to avoid log(0)
                         sr=sr, hop_length=16, x_axis='time', y_axis='mel', fmin=4000, fmax=11000)
plt.title("Dominant Frequencies Only")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
# Save to current working directory
plt.savefig("pl5_dominantfrequency_spectrogram.png") 

### APPLY MASK ##

# Inspect the array to see if there are regions of 0s causing the border
print(f"Min value of S_dB_after_masked: {S_dB.min()}")
print(f"Max value of S_dB_after_masked: {S_dB.max()}")

# Apply close to the binary mask
closed_mask = binary_closing(S_dB, structure=np.ones((3, 3)))

# Label the connected pixels in the image
labels = label(closed_mask)

# Set the minimum size of the regions to keep
# User defined in Lotti App so AJ replaced with usual region size
min_size = 25  
# selected_size = self.region_size_combobox.get()
# if not selected_size:
#         min_size = 50
# else:
#         min_size = int(selected_size)

# Iterate over the regions in the labeled image
for region in regionprops(labels):
    # Get the size of the region
    size = region.area
    # Remove the region if its size is smaller than the minimum size
    if size <= min_size:
        minr, minc, maxr, maxc = region.bbox
        closed_mask[minr:maxr, minc:maxc] = 0

# Apply the cleaned mask to the original spectrogram
cleaned_spectrogram = S_dB_norm * closed_mask

# Plot the cleaned spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(cleaned_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar()
plt.title("Cleaned Spectrogram After Morphological Filtering")
plt.tight_layout()
plt.savefig("pl6_cleaned_spectrogram.png")  # Saves to the working directory

#Summarise
print(f"CLEANED SPEC:")
print(f"Spec Min: {cleaned_spectrogram.min()}, Max: {cleaned_spectrogram.max()}\n")
print(f"Top 5 values: {cleaned_spectrogram[:5]}\n")
print(f"Bottom 5 values: {cleaned_spectrogram[-5:]}\n")

### CONTOURS ###

# Add frequency contours
# Original code to use values in lotti app
#contour_number = self.contour_no_combobox.get()
#if not contour_number:
#        num_contours = 10
#else:
#        num_contours = int(contour_number)
#
#min_contour_level = self.contour_min_combobox.get()
#if not min_contour_level:
#        min_level = 0.4
#else:
#        min_level = float(min_contour_level)


#Alternative code to use manually set values (replaces Lotti App)
num_contours = 25
min_level = 0.1
contour_levels = np.linspace(S_dB_norm.min() + (S_dB_norm.max() - S_dB_norm.min()) * min_level, S_dB_norm.max(), num_contours)

# Create a single array that represents the contour plot
contour_array = np.zeros_like(cleaned_spectrogram)

for i in range(num_contours - 1):
    contour_array[(cleaned_spectrogram >= contour_levels[i]) & (cleaned_spectrogram <= contour_levels[i + 1])] = i

sums = np.sum(contour_array, axis = 0)
peaks, _ = find_peaks(sums) 

# ensure that there is no failure by creating a fail safe if no peaks are detected
if len(peaks) > 0: 
    call_start = max(np.min(peaks[0]) - 40, 0)  # prevent negative
else:
    call_start = 1 # give a little bit of space before the call

# Compute frame energy (sum across frequency bins for each time frame)
frame_energy = np.sum(S_dB_norm, axis=0)

# Optional smoothing (can help with noise)
# frame_energy = np.convolve(frame_energy, np.ones(5)/5, mode='same')

# Plot the frame energy
plt.figure(figsize=(12, 4))
plt.plot(frame_energy, label='Frame Energy')
plt.axvline(call_start, color='green', linestyle='--', label='Call Start')
plt.title('Frame Energy Over Time')
plt.xlabel('Frame Index')
plt.ylabel('Summed Energy (across frequencies)')
plt.legend()
plt.tight_layout()
plt.savefig("pl7_frame_energy.png")  # Saves to the working directory

# Parameters
silence_threshold = 2  # energy below this is considered silence
min_blank_duration = 5  # number of consecutive frames to be considered a blank
min_duration_before_end_check = 200  # number of frames to wait after call_start before looking for end

# Compute the first frame index to begin looking for silence
check_start_frame = call_start + min_duration_before_end_check

# Ensure we don't exceed the frame range
if check_start_frame >= len(frame_energy):
    call_end_frame = len(frame_energy)
else:
    # Loop through each frame from check_start_frame onward
    for i in range(check_start_frame, len(frame_energy) - min_blank_duration):
        if np.all(frame_energy[i:i + min_blank_duration] < silence_threshold):
            call_end_frame = i
            break
    else:
        call_end_frame = len(frame_energy)  # fallback: take the rest

# get the time of the start
    # Convert frame counts to time (seconds).
start_time = librosa.frames_to_time(call_start, sr = sr, hop_length = 16)
#AJ increased this from 0.1 to + 0.11
#end_times = start_time + 0.11
end_times = librosa.frames_to_time(call_end_frame, sr=sr, hop_length=16)
end_frames = librosa.time_to_frames(end_times, sr = sr, hop_length = 16)
# Select the section of the Mel spectrogram corresponding to x and x + 200ms (AJ changed not sure about new value)
S_dB_section = contour_array[: , call_start:end_frames]

#Summarise
print(f"call_start: {call_start}")
print(f"call_end_frame: {call_end_frame}")
print(f"end_frames: {end_frames}")
print(f"S_dB_section shape: {S_dB_section.shape}")
print(f"contour_array shape: {contour_array.shape}")
print(f"frame_energy length: {len(frame_energy)}")

#Plot energy again
plt.figure(figsize=(10, 3))
plt.plot(frame_energy, color='black')
plt.axhline(y=silence_threshold, color='red', linestyle='--', label='Silence Threshold')
plt.axvline(x=call_start, color='green', linestyle='--', label='Call Start')
plt.axvline(x=call_end_frame, color='blue', linestyle='--', label='Call End')
plt.title("Frame Energy with Call Boundaries")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.legend()
plt.tight_layout()
plt.savefig("pl9_frame_energy_debug.png", dpi=300)

#Plot
# Plot the extracted section of the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB_section, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
plt.title("Extracted Section of Mel Spectrogram")
plt.colorbar()
plt.tight_layout()
# Save the plot to file (optional)
plt.savefig("pl8_extracted_spectrogram_section.png", dpi=300)

#Summarise
print(f"EXTRACTED SPEC:")
print(f"Spec Min: {S_dB_section.min()}, Max: {S_dB_section.max()}\n")
print(f"Top 5 values: {S_dB_section[:5]}\n")
print(f"Bottom 5 values: {S_dB_section[-5:]}\n")