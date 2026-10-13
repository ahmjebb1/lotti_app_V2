### IMPORTS ###

import os
import time
import tkinter as tk
from tkinter import Frame, StringVar, Label, Entry, Button, Toplevel, filedialog, messagebox, ttk, LabelFrame
import pandas as pd
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import find_peaks
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa.display

### DIRECTORIES ###

#Find out working directory
print("Current working directory:", os.getcwd())

#set folder to save outputs
output_folder = os.getcwd()

### LOADING ###

#Testing librosa.load
#Set path of audio file
audio_path = "C:/Users/bi1ahj/Desktop/Selection_Tables/250210_ML50_churr_dataset/Dataset/2023/BNL/Audio Data/230403_Nest_29_R2.wav" 
#Testing with a random section of a random file (may need to alter offset dependent on churr focussing on)
original_y, sr = librosa.load(audio_path, offset = 571.1657, duration = 0.13, sr = 48000)

#Apply a fade-in to remove artefacts introduced when slicing
#Calculates how many samples are equivelent to 5ms
fade_len = int(0.005 * sr)
#Creates a linearly increasing array from 0 to 1 represents a volume envelope that ramps up from silence to full volume.
#Applies the fade in to only the length = fade len
#Each of the samples is multiplied by its corresponding fade value. 
original_y[:fade_len] *= np.linspace(0, 1, fade_len)

# Compute the Mel spectrogram (Not originally in lotti App)
original_S = librosa.feature.melspectrogram(y = original_y, sr=sr, n_fft=768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

# Plot the original spectrogram cleanly
plt.figure(figsize=(7, 5))
librosa.display.specshow(original_S, sr=sr, x_axis=None, y_axis=None, cmap="Greys")

# Remove all axes, ticks, labels, and frames
plt.axis('off')
plt.gca().set_frame_on(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# Tight layout with no padding or whitespace
plt.tight_layout(pad=0)

# Save clean, borderless, transparent figure
plt.savefig(
    os.path.join(output_folder, "pl1_paper_original_spec.png"),
    dpi=300,
    bbox_inches='tight',
    pad_inches=0,
    transparent=True
)

plt.close()

### NOISE REDUCTION ###

# Specify onset time and duration (in seconds) for noise sample
noise_onset_sec = 0.8     # e.g., start at 0.4 seconds
noise_duration_sec = 0.2  # e.g., use 0.2 seconds of noise

# Convert to sample indices
start_sample = int(noise_onset_sec * sr)
end_sample = int((noise_onset_sec + noise_duration_sec) * sr)

# Extract the noise clip
noise_clip, _ = librosa.load(audio_path, offset = noise_onset_sec, duration = noise_duration_sec, sr=sr)

# Apply stationary noise reduction
y_stationary = nr.reduce_noise(
    y=original_y,
    y_noise=noise_clip,
    hop_length=16,
    sr=sr,
    n_fft=768,
    stationary=True,
    n_std_thresh_stationary = 1
)

# Then apply non-stationary reduction
y = nr.reduce_noise(
    y=y_stationary,
    sr=sr,
    hop_length=16,
    n_fft=768,
    thresh_n_mult_nonstationary = 12
)

#Run the noise reduction
#y = nr.reduce_noise(y=original_y, sr=sr, thresh_n_mult_nonstationary=12, n_fft=768)

# Compute the Mel spectrogram for the denoised signal
S = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

### DB SPEC ###
S_dB = librosa.amplitude_to_db(S, ref=np.max)

# Save the spectrogram to the working directory
plt.savefig(os.path.join(output_folder, "pl3_paper_dB.png"))
plt.close()

### NORMALIZE ###

# Normalize the array to the range [0, 1]
#S_dB_norm = librosa.util.normalize(S_dB, axis=0)
# Manually scale to [0, 1]
S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

### NEW SPEC OF DOM FREQ ###

# Find the index of the frequency with the highest magnitude at each time point
dominant_indices = np.argmax(S, axis=0)

# Create a new spectrogram with only the dominant frequencies set to their original amplitude
S_dominant = np.zeros(S.shape)

# Create a 2D array of indices for proper indexing
time_indices = np.arange(S.shape[1])
indices_array = np.stack((dominant_indices, time_indices), axis=-1)
S_dominant[tuple(indices_array.T)] = S[tuple(indices_array.T)]

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

# Plot the extracted section of the spectrogram
plt.figure(figsize=(7, 5))
librosa.display.specshow(S_dB_section, sr=sr, x_axis=None, y_axis=None, cmap="Greys")

# Remove all axes, ticks, and labels
plt.axis('off')

# Remove any spines, borders, or backgrounds
plt.gca().set_frame_on(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# No title, colorbar, or padding
plt.tight_layout(pad=0)

# Save the figure with transparent background (optional)
plt.savefig("pl9_paper_extracted_spectrogram_section.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

# Close the plot to free memory
plt.close()
