############################################ IMPORTS ##############################################

import os
import numpy as np
import noisereduce as nr

from scipy.signal import find_peaks
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import librosa
import librosa.display

############################################ DIRECTORIES ##############################################

#Find out working directory
print("Current working directory:", os.getcwd())

#set folder to save outputs
output_folder = os.getcwd()

############################################## LOADING ################################################

#Set path to audio
audio_path = "C:/Users/bi1ahj/Desktop/Selection_Tables/250210_ML50_churr_dataset/Dataset/2024/BOR/Audio Data/240331_Nest_43_R1.wav" 

#Select specific churr to load
original_ch, sr = librosa.load(audio_path, offset = 252.1189, duration = 0.3, sr = 48000)

#Select specific tweep to load
original_y, sr = librosa.load(audio_path, offset = 252.1189, duration = 0.13, sr = 48000)

######################################### TWEEP PROCESSING ##############################################

### FADE-IN & CONVERT TO SPECTROGRAM ###

# Convert full churr to Mel spectrogram
original_S_ch = librosa.feature.melspectrogram(y = original_ch, sr=sr, n_fft=768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

#Apply a fade-in to remove artefacts introduced when slicing
#Calculates how many samples are equivelent to 5ms
fade_len = int(0.005 * sr)
#Creates a linearly increasing array from 0 to 1 represents a volume envelope that ramps up from silence to full volume.
#Applies the fade in to only the length = fade len
#Each of the samples is multiplied by its corresponding fade value. 
original_y[:fade_len] *= np.linspace(0, 1, fade_len)

# Compute the tweep Mel spectrogram
original_S = librosa.feature.melspectrogram(y = original_y, sr=sr, n_fft=768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

### NOISE REDUCTION ###

# Specify onset time and duration (in seconds) for noise sample
noise_onset_sec = 2.0     # e.g., start at 0.4 seconds
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
    thresh_n_mult_nonstationary = 8
)

# Compute the Mel spectrogram for the denoised signal
S = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

### DB SPEC ###

S_dB = librosa.amplitude_to_db(S, ref=np.max)

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

### CONTOURS ###

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

# Parameters
silence_threshold = 5  # energy below this is considered silence
min_blank_duration = 4 # number of consecutive frames to be considered a blank
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

############################################ PLOTTING ##########################################################

### CHECK THE MIN MAX VALUES OF EACH SPECTROGRAM
print(np.min(original_S_ch), np.max(original_S_ch)) #1.997071e-10 0.018654864
print(np.min(original_S), np.max(original_S)) #1.997071e-10 0.018654864
print(np.min(S), np.max(S)) #1.3461817e-14 0.013944434
print(np.min(S_dB_norm), np.max(S_dB_norm)) #0.0 1.0
print(np.min(S_dominant), np.max(S_dominant)) #0.0 0.013944434002041817
print(np.min(cleaned_spectrogram), np.max(cleaned_spectrogram)) #0.0 1.0
print(np.min(S_dB_section), np.max(S_dB_section)) #0.0 23.0

### DEFINE PERSONALISED HEAT_MAP

# Define colours
colours = ["#2b2627", "#4477AA", "#BBBBBB", "#FFFFFF", "#CCBB44", "#EE6677", "#AA3377"]

# Create heatmap
custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", colours)

### SET UP PANEL PLOT

fig = plt.figure(figsize=(12, 8), facecolor="white")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

### PANEL 1 ###
ax1 = fig.add_subplot(gs[0, 0])
librosa.display.specshow(original_S_ch, x_axis='time', y_axis='mel', sr=sr, ax=ax1, cmap=custom_cmap,
                         vmin=0)

ax1.set_title("A) Original Churr")

### PANEL 2 ###
ax2 = fig.add_subplot(gs[0, 1])
librosa.display.specshow(original_S, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap=custom_cmap,
                         vmin=0)

ax2.set_title("B) Segmented Tweep")

### PANEL 3 ###
ax3 = fig.add_subplot(gs[0, 2])
librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, ax=ax3, cmap=custom_cmap,
                         vmin=0)

ax3.set_title("C) Denoised Spectrogram")

### PANEL 4 ###
ax4 = fig.add_subplot(gs[1, 0])  
librosa.display.specshow(S_dB_norm, x_axis='time', y_axis='mel', sr=sr, ax=ax4, cmap=custom_cmap,
                         vmin=0)

ax4.set_title("D) Normalized Spectrogram")

### PANEL 5 ###
#ax5 = fig.add_subplot(gs[1, 1])
#librosa.display.specshow(librosa.amplitude_to_db(S_dominant + 1e-10, ref=np.max),  # add small value to avoid log(0)
#                         sr=sr, hop_length=16, x_axis='time', y_axis='mel', ax=ax5, cmap=custom_cmap,
#                         vmin=0)  
#
#ax5.set_title("5) Dominant Frequency Spectrogram")
#ax5.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

### PANEL 6 ###
ax5 = fig.add_subplot(gs[1, 1])  
librosa.display.specshow(cleaned_spectrogram, x_axis='time', y_axis='mel', sr=sr, ax=ax5, cmap=custom_cmap,
                         vmin=0)

ax5.set_title("E) Cleaned Spectrogram")

### PANEL 7 ###
ax6 = fig.add_subplot(gs[1, 2])  

# Main line
ax6.plot(frame_energy, color="#2b2627")

# Silence Threshold
ax6.axhline(y=silence_threshold, color="#CCBB44", linestyle='--', label='Silence Threshold')
ax6.text(y = max(frame_energy) * 0.95, x = call_start + 10,  s = "Call Start", va="bottom", ha="left", fontsize=7)

# Call Start
ax6.axvline(x=call_start, color="#4477AA", linestyle='--', label='Call Start')
ax6.text(y = silence_threshold, x = len(frame_energy) * 0.25, s = "Silence Threshold", va="bottom", ha="left", fontsize=7)

# Call End
ax6.axvline(x=call_end_frame, color="#EE6677", linestyle='--', label='Call End')
ax6.text(y = max(frame_energy) * 0.95, x = call_end_frame - 10, s = "Call End", va="bottom", ha="right", fontsize=7)

ax6.set_xlabel("Frame")
ax6.set_ylabel("Energy")

ax6.set_title("F) Frame Energy")

### PANEL 8 ###
ax7 = fig.add_subplot(gs[2, 0])  
librosa.display.specshow(S_dB_section, x_axis='time', y_axis='mel', sr=sr, ax=ax7, cmap=custom_cmap,
                         vmin=0)

ax7.set_title("G) Final Spectrogram")

# Adjust and show
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "pl1_mel_spectrogram_original.png"))
plt.close()



##Plot
## Set up plot
#plt.figure(figsize=(10, 6))
## Plot dominant-frequency-only spectrogram
#librosa.display.specshow(librosa.amplitude_to_db(S_dominant + 1e-10, ref=np.max),  # add small value to avoid log(0)
#                         sr=sr, hop_length=16, x_axis='time', y_axis='mel', fmin=4000, fmax=11000)
#plt.title("Dominant Frequencies Only")
#plt.colorbar(format="%+2.0f dB")
#plt.tight_layout()
## Save to current working directory
#plt.savefig("pl5_dominantfrequency_spectrogram.png") 
#
## Plot the cleaned spectrogram
#plt.figure(figsize=(10, 4))
#librosa.display.specshow(cleaned_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
#plt.colorbar()
#plt.title("Cleaned Spectrogram After Morphological Filtering")
#plt.tight_layout()
#plt.savefig("pl6_cleaned_spectrogram.png")  # Saves to the working directory
#
#
#
##Plot energy again
#plt.figure(figsize=(10, 3))
#plt.plot(frame_energy, color='black')
#plt.axhline(y=silence_threshold, color='red', linestyle='--', label='Silence Threshold')
#plt.axvline(x=call_start, color='green', linestyle='--', label='Call Start')
#plt.axvline(x=call_end_frame, color='blue', linestyle='--', label='Call End')
#plt.title("Frame Energy with Call Boundaries")
#plt.xlabel("Frame")
#plt.ylabel("Energy")
#plt.legend()
#plt.tight_layout()
#plt.savefig("pl9_frame_energy_debug.png", dpi=300)
