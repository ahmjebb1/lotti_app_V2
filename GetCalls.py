import os
import logging
import traceback
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
from FilterCallsPage import FilterCallsPage

#Logging an error message plus the full traceback to help identify where and how errors occur
logging.basicConfig(
    filename="spectrogram_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(message):
    logging.info(message)

def log_error(error):
    logging.error(error)
    logging.error(traceback.format_exc())

#Move onto Lotti processing code
class GetCallsPage(Frame):
    def __init__(self, parent, controller):
            Frame.__init__(self, parent)

            # Configure column and row weights        
            self.image_popup = None  # Reference to the popup window`

            # Initialize variables
            self.folder_path = ""
            self.data_file_path = StringVar()

            meta_frame = LabelFrame(self, text = "Data")
            meta_frame.grid(row = 1, column=0, sticky="nsew", padx=5, pady=5)
            meta_frame.columnconfigure(1, weight=1)
            meta_frame.columnconfigure(2, weight=1) 
            meta_frame.columnconfigure(3, weight=1)           

            # Meta data display (non-editable entry)
            self.set_text_meta = Label(meta_frame, text = "Select data")
            self.set_text_meta.grid(row = 1, column= 1, sticky= "nsew", padx=5, pady=5)
            self.button_6 = Button(meta_frame, text= "Select", command=lambda: (print("Uploaded data and save"), self.upload_and_save_csv()))
            self.button_6.grid(row = 1, column = 3, sticky= "nsew", padx=5, pady=5)
            self.meta_data_var = StringVar()
            self.meta_data_entry = Entry(meta_frame, textvariable=self.meta_data_var, state='readonly')
            self.meta_data_entry.grid(row=1, column=2, sticky= "nsew", padx=5, pady=5)

            # get the audio file path to display in box
            self.set_text_audio = Label(meta_frame, text = "Select audio folder")
            self.set_text_audio.grid(row = 2, column= 1, sticky= "nsew", padx=5, pady=5)
            self.button_7 = Button(meta_frame, text="Select", command=lambda: (print("button_7 clicked"), self.set_folder_path()))
            self.button_7.grid(row = 2, column=3, sticky= "nsew", padx=5, pady=5)
            self.audiopath_var = StringVar()
            self.audiopath_entry = Entry(meta_frame, textvariable=self.audiopath_var, state='readonly')
            self.audiopath_entry.grid(row=2, column=2, sticky= "nsew", padx=5, pady=5)

            ### Variable Frame ###
            self.variable_frame = LabelFrame(self, text = "Variables")
            self.variable_frame.grid(row = 2, column=0, sticky="nsew", padx=5, pady=5)
            self.variable_frame.columnconfigure(1, weight=1)
            self.variable_frame.columnconfigure(2, weight=1)

            # Call type dropdown menu
            self.set_text = Label(self.variable_frame, text = "Select call type")
            self.set_text.grid(row = 1, column= 1, sticky= "nsew", padx=5, pady=5)
            options = ["Ch", "T", "P", "M", "B"]
            self.selected_code = ttk.Combobox(self.variable_frame, values=options, state="readonly")
            self.selected_code.set(options[0])  # default value
            self.selected_code.grid(row=1, column=2, sticky= "nsew", padx=5, pady=5)
                                
            # Combobox for contour number
            self.set_text_contour_no = Label(self.variable_frame, text = "Select contour no.")
            self.set_text_contour_no.grid(row = 2, column= 1, sticky= "nsew", padx=5, pady=5)
            options_contour_no = [5, 10, 15, 20, 25]
            self.contour_no_combobox = ttk.Combobox(self.variable_frame, values=options_contour_no, state="readonly")
            self.contour_no_combobox.set(options_contour_no[1])  # default value
            self.contour_no_combobox.grid(row=2, column=2, sticky="nwse", padx=5, pady=5)

            # Combobox for contour minimum
            self.set_text_contour_min = Label(self.variable_frame, text = "Select min contour")
            self.set_text_contour_min.grid(row = 3, column= 1, sticky= "nsew", padx=5, pady=5)
            options_contour_min = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            self.contour_min_combobox = ttk.Combobox(self.variable_frame, values=options_contour_min, state="readonly")
            self.contour_min_combobox.set(options_contour_min[3])  # default value
            self.contour_min_combobox.grid(row=3, column=2, sticky="nwew", padx=5, pady=5)

            # Combobox for region size
            self.set_text_region = Label(self.variable_frame, text = "Select region size")
            self.set_text_region.grid(row = 4, column= 1, sticky= "nsew", padx=5, pady=5)
            options_region_size = [25, 50, 75, 100, 125, 150]
            self.region_size_combobox = ttk.Combobox(self.variable_frame, values=options_region_size, state="readonly")
            self.region_size_combobox.set(options_region_size[1])  # default value
            self.region_size_combobox.grid(row=4, column=2, sticky="nsew", padx=5, pady=5)
          
            # Text box for noise onset
            self.set_text_onset = Label(self.variable_frame, text = "Select noise onset (s.ms)")
            self.set_text_onset.grid(row=5, column=1, sticky="nsew", padx=5, pady=5)
            self.noise_onset_var = StringVar()
            self.noise_onset_entry = Entry(self.variable_frame, textvariable=self.noise_onset_var)
            self.noise_onset_entry.grid(row=5, column=2, sticky="nsew", padx=5, pady=5)

            ## Text box for colour ring choice
            self.set_text_cr = Label(self.variable_frame, text = "Select colour ring of interest")
            self.set_text_cr.grid(row=6, column=1, sticky="nsew", padx=5, pady=5)
            self.cr_var = StringVar()
            self.cr_entry = Entry(self.variable_frame, textvariable=self.cr_var)
            self.cr_entry.grid(row=6, column=2, sticky="nsew", padx=5, pady=5)

            # Action button for 'run'
            self.button_8 = Button(self,text = "Run",command=lambda: (print("button_8 clicked"), self.button_8_click_get_calls()))
            self.button_8.grid(row = 3, column=0, sticky= "nsew", padx=5, pady=5, columnspan= 3)            

            # Action button for 'filter calls'
            self.button_9 = Button(self,text = "Filter calls",command=lambda: (print("button_9 clicked"), self.open_popup()))
            self.button_9.grid(row = 4, column=0, sticky= "nsew", padx=5, pady=5, columnspan= 3)            

    def open_popup(self):
                    # Create a popup window
                    self.image_popup = tk.Toplevel(self)
                    self.filter_calls = FilterCallsPage(self.image_popup)

                    # Create a frame for the image
                    image_frame = ttk.Frame(self.image_popup)
                    image_frame.pack(fill="both", expand=True)

                    # Create a canvas inside the image frame
                    canvas = tk.Canvas(image_frame)
                    canvas.pack(fill="both", expand=True)

                    # Create the canvas image object as an attribute of the class
                    canvas_image = canvas.create_image(0, 0, anchor="nw", image=None)

                    # Create a frame for buttons
                    button_frame = ttk.Frame(self.image_popup)
                    button_frame.pack()

                    # Place buttons on the grid in the button frame
                    button_1 = ttk.Button(button_frame, text="Remove", command=self.filter_calls.remove_current_image)
                    button_2 = ttk.Button(button_frame, text="Previous", command=lambda: self.filter_calls.display_image("-", canvas, canvas_image))
                    button_3 = ttk.Button(button_frame, text="Next", command=lambda: self.filter_calls.display_image("+", canvas, canvas_image))
                    button_4 = ttk.Button(button_frame, text="Done", command=self.filter_calls.filter_high_quality)

                    button_1.grid(row=0, column=0, padx=5, pady=5)
                    button_2.grid(row=0, column=1, padx=5, pady=5)
                    button_3.grid(row=0, column=2, padx=5, pady=5)
                    button_4.grid(row=0, column=3, padx=5, pady=5)

                    # Display the first image in the popup
                    self.filter_calls.display_image(direction=None, canvas=canvas, canvas_image=canvas_image)

    def open_csv_file(self, file_path):
                data_file = pd.read_csv(file_path)
                
                return data_file

            # Define the function to set the folder path
    def set_folder_path(self):
                folder_selected = filedialog.askdirectory()
                if folder_selected:
                    global folder_path
                    folder_path = folder_selected + '/'
                    print(folder_path)
                    self.audiopath_var.set(folder_path)

            # upload csv and save to new folder

    def upload_and_save_csv(self):
                file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
                if file_path:
                    print(f"Selected file: {file_path}")
                    file_name = os.path.basename(file_path)
                    name = os.path.basename(file_path).split('.')[0]

                    #Save the name for later use
                    self.name = name

                    # Read the file into a pandas DataFrame
                    df = pd.read_csv(file_path, delimiter='\t')  # or adjust the delimiter as needed

                    # Split the 'Annotation' column
                    if 'Annotation' in df.columns:
                        split_data = df['Annotation'].str.split('_', expand=True)
                        df['call_type'] = split_data[0]
                        df['ID'] = split_data[1]
                        df['sound.files'] = f'{name}.WAV'
                        df['start'] = df['Begin Time (s)']

                    # Create a new folder and save the modified DataFrame as a CSV
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    new_folder_path = os.path.join(script_dir, file_name.rstrip('.csv') + '_data')
                    os.makedirs(new_folder_path, exist_ok=True)

                    new_file_path = os.path.join(new_folder_path, file_name)
                    df.to_csv(new_file_path, index=False)
                    print(f"File saved to: {new_file_path}")

                    self.meta_data_var.set(file_name)
                    self.data_file_path.set(new_file_path)

                    # Optionally update other variables or return new file path
                    return new_file_path
                else:
                    self.name = None
                    return None

    # these are the functions to extract the tweep from a Lotti churr call  
    def get_tweep(self, call, folder_path, start, call_id, selec):
        try:
                                
                                # Only process if call_id matches colour ring input
                                if call_id != self.cr_var.get():
                                    return None, None
                                
                                full_id = f"{call_id}_{selec}"
                                # log a message in the logging file listing the call details processed and the call onset
                                log_message(f"\nProcessing: {call} | ID: {full_id} | Start: {start:.3f}")
        
                                #offset = start -> starts reading after a select time named "start".
                                #duration -> loading up this much audio in seconds.
                                #sr -> target sampling rate
                                #y, sr = librosa.load(folder_path + call, offset = start, duration = 0.13, sr = 48000)
                                y_stereo, sr = librosa.load(folder_path + call, offset = start, duration = 0.13, sr = 48000, mono=False)

                                if y_stereo.ndim == 1:
                                    y = y_stereo  # mono
                                else:
                                    log_message(f"Stereo audio detected in: {os.path.basename(folder_path)} — using channel 1 only")
                                    y = y_stereo[0]  # channel 1 of stereo

                                # log a message related to check if y loaded correctly
                                log_message(f"Raw audio length: {len(y)}, max: {np.max(y):.5f}, min: {np.min(y):.5f}")

                                #Apply a fade-in to remove artefacts introduced when slicing
                                #Calculates how many samples are equivelent to 5ms
                                fade_len = int(0.005 * sr)
                                #Creates a linearly increasing array from 0 to 1 represents a volume envelope that ramps up from silence to full volume.
                                #Applies the fade in to only the length = fade len
                                #Each of the samples is multiplied by its corresponding fade value. 
                                y[:fade_len] *= np.linspace(0, 1, fade_len)
                                y[-fade_len:] *= np.linspace(1, 0, fade_len)

                                # Noisereduce
                                # Specify onset time and duration (in seconds) for noise sample
                                # Retrieve from Lotti App inputs

                                noise_onset_str = self.noise_onset_var.get()
                                       
                                if not noise_onset_str.strip():
                                    log_error(f" - Error Missing noise onset time for ID: {full_id}")
                                    return None, None

                                # make sure noise onset is a float
                                noise_onset_sec = float(noise_onset_str)

                                # always the same value
                                noise_duration_sec = 0.2  

                                # Extract the noise clip & set audio file to extract noise from
                                wav_filename = f"{self.name}.wav"
                                noise_clip, _ = librosa.load(os.path.join(folder_path, wav_filename), offset = noise_onset_sec, duration = noise_duration_sec, sr=sr)

                                # Apply stationary noise reduction
                                y_stationary = nr.reduce_noise(
                                    y = y,
                                    y_noise=noise_clip,
                                    hop_length=16,
                                    sr=sr,
                                    n_fft=768,
                                    stationary=True,
                                    n_std_thresh_stationary = 1.5
                                )

                                # Then apply non-stationary reduction
                                y_nstationary = nr.reduce_noise(
                                    y = y_stationary,
                                    sr=sr,
                                    hop_length=16,
                                    n_fft=768,
                                    thresh_n_mult_nonstationary = 8
                                )

                                # Log information after noise processing
                                log_message(f"Post-noise-reduction max: {np.max(y_nstationary):.5f}, min: {np.min(y_nstationary):.5f}")

                                # Compute the Mel spectrogram
                                    # computing a mel spectrogram using a time-series input
                                    # sr = sampling rate
                                    # hop_length: number of samples between successive frames.
                                    # n_mels:number of Mel bands to generate, 
                                    # fmin: lowest frequency (in Hz) 
                                    # fmax: highest frequency (in Hz)
                                S = librosa.feature.melspectrogram(y = y_nstationary, sr=sr, n_fft = 768, hop_length=16, n_mels=128, fmin=4000, fmax=11000)

                                # Convert to magnitude spectrogram (dB)
                                    #Convert an amplitude spectrogram to dB-scaled spectrogram
                                    # ref = If scalar, the amplitude abs(S) is scaled relative to ref: 20 * log10(S / ref).
                                    # Zeros in the output correspond to positions where S == ref.
                                S_dB = librosa.amplitude_to_db(S, ref=np.max)

                                # Normalize the array to the range [0, 1]
                                # Manually scale to [0, 1]
                                S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

                                # Log after normalizing the spectrogram and converting it
                                log_message(f"S_dB min: {S_dB.min():.2f}, max: {S_dB.max():.2f}")

                                # Find the index of the frequency with the highest magnitude at each time point
                                dominant_indices = np.argmax(S, axis=0)

                                # Create a new spectrogram with only the dominant frequencies set to their original amplitude
                                S_dominant = np.zeros(S.shape)

                                # Create a 2D array of indices for proper indexing
                                time_indices = np.arange(S.shape[1])
                                indices_array = np.stack((dominant_indices, time_indices), axis=-1)

                                S_dominant[tuple(indices_array.T)] = S[tuple(indices_array.T)]

                                # Apply close to the binary mask
                                closed_mask = binary_closing(S_dB, structure=np.ones((3, 3)))

                                # Label the connected pixels in the image
                                labels = label(closed_mask)

                                # Set the minimum size of the regions to keep
                                selected_size = self.region_size_combobox.get()

                                if not selected_size.strip():
                                    log_error(f" - Error  Missing region size for ID: {full_id}")
                                    return None, None
                                min_size = int(selected_size)

                                for region in regionprops(labels):
                                    if region.area <= min_size:
                                        minr, minc, maxr, maxc = region.bbox
                                        closed_mask[minr:maxr, minc:maxc] = 0

                                # Apply the cleaned mask to the original spectrogram
                                cleaned_spectrogram = S_dB_norm * closed_mask

                                # Add frequency contours
                                contour_number = self.contour_no_combobox.get()
                                
                                if not contour_number.strip():
                                    log_error(f" - Error ❌ Missing contour number for ID: {full_id}")
                                    return None, None
                                num_contours = int(contour_number)

                                # Contour minimum level validation
                                min_contour_level = self.contour_min_combobox.get()
                                if not min_contour_level.strip():
                                    log_error(f" - Error ❌ Missing minimum contour level for ID: {full_id}")
                                    return None, None
                                min_level = float(min_contour_level)                            
                                
                                contour_levels = np.linspace(S_dB_norm.min() + (S_dB_norm.max() - S_dB_norm.min()) * min_level, S_dB_norm.max(), num_contours)

                                # Create a single array that represents the contour plot
                                contour_array = np.zeros_like(cleaned_spectrogram)

                                for i in range(num_contours - 1):
                                    contour_array[(cleaned_spectrogram >= contour_levels[i]) & (cleaned_spectrogram <= contour_levels[i + 1])] = i

                                sums = np.sum(contour_array, axis = 0)

                                peaks, _ = find_peaks(sums) 

                                # Ensure that there is no failure by creating a fail safe if no peaks are detected
                                call_start = int(max(np.min(peaks) - 40, 0)) if len(peaks) > 0 else 1
                                
                                # Compute frame energy (sum across frequency bins for each time frame)
                                frame_energy = np.sum(S_dB_norm, axis=0)

                                # Energy parameters
                                silence_threshold = 5  # energy below this is considered silence
                                min_blank_duration = 4  # number of consecutive frames to be considered a blank
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
                                # end_times
                                end_times = librosa.frames_to_time(call_end_frame, sr=sr, hop_length = 16)
                                end_frames = librosa.time_to_frames(end_times, sr = sr, hop_length = 16)
                                # Select the section of the Mel spectrogram corresponding to x and x + 200ms 
                                S_dB_section = contour_array[: , call_start:end_frames]
                             
                                #log message after trimming the final spectrogram product
                                log_message(f"call_start: {call_start}, end_frames: {end_frames}")
                                #call_id = f'{call_id}_{selec}'

                                log_message(f" - Success (shape: {S_dB_section.shape})")
                                
                                return S_dB_section, full_id
    
        except Exception as e:
                                log_error(f" - Error while processing {call_id}_{selec}: {e}")
                                return None, None       
    
    def save_to_pickle_get_calls(self, meta_data_var_label, X_calls, X_file_name, Y_calls, Y_file_name):
                    # Create the folder if it doesn't exist
                    output_folder = os.path.join(os.getcwd(), meta_data_var_label)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Save the pickle files in the created folder
                    with open(os.path.join(output_folder, X_file_name), 'wb') as f:
                        pickle.dump(X_calls, f)

                    with open(os.path.join(output_folder, Y_file_name), 'wb') as f:
                        pickle.dump(Y_calls, f)

                    print(f"Data saved to {output_folder}")
    
    def save_images(self, images, filenames, meta_data_var_label):
                    """
                    Save an array of images as PNG files.

                    Args:
                        images (numpy.ndarray): An array of images with shape (num_images, height, width, channels).
                        filenames (list): A list of filenames to use for each image.

                    Raises:
                        ValueError: If the length of `filenames` does not match the number of images in `images`.
                    """
                    #if len(images) != len(filenames):
                    #    raise ValueError("The length of `filenames` must match the number of images in `images`.")
                    
                    # Create the folder if it doesn't exist
                    output_folder = os.path.join(os.getcwd(), f"{meta_data_var_label}_images")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Debugging- only log debug info for the first image (if any exist)
                    if len(images) > 0:
                        img = images[0]  # Process only the first spectrogram
                        sanitized_filename = re.sub(r'[<>:"/\\|?*]', '', filenames[0])
                                          # Open debug log file
                        debug_file_path = os.path.join(output_folder, "debug_log.txt")
                        with open(debug_file_path, 'w') as debug_file:
                            debug_file.write(f"\nDEBUG - {sanitized_filename}:\n")
                            debug_file.write(f"Top 5 rows min/max: {[ (row.min(), row.max()) for row in img[:5] ]}\n")
                            debug_file.write(f"Bottom 5 rows min/max: {[ (row.min(), row.max()) for row in img[-5:] ]}\n")
                            debug_file.write(f"Overall min/max: {img.min()}, {img.max()}\n")
                            
                            #check file name
                            debug_file.write(f"{self.name}\n")
                                            
                            zero_rows = np.where(np.all(np.isclose(img, 0, atol=1e-6), axis=1))[0]
                            debug_file.write(f"Fully zero Mel bands (row indices): {zero_rows}\n")
                                                                                     
                    for i, img in enumerate(images):
                        sanitized_filename = re.sub(r'[<>:"/\\|?*]', '', filenames[i])  
                        #AJ is this potentially where the plot dimensions are set? C
                        #Changed figsize = (10,5) to (5,5)
                        plt.figure(figsize=(5, 5))
                        plt.imshow(img, origin='lower', aspect='auto', cmap='gray_r')
                        plt.xlabel('Time (frames)')
                        plt.ylabel('Frequency (Hz)')
                        plt.title(filenames[i])
                        plt.tight_layout()
                        # Save images in the created folder
                        plt.savefig(os.path.join(output_folder, sanitized_filename + '.png'))
                        plt.close()

                    print(f"All images saved to {output_folder}")
                  

    def extract_tweeps(self, data, folder_path, type, file_name):
                # Create a popup window for the progress bar
                popup = Toplevel()
                popup.title("Progress")
                popup.geometry("300x100") 
                
                progress_bar = ttk.Progressbar(popup,orient="horizontal", mode="determinate")
                progress_bar.pack(expand=True, fill='x', padx=20, pady=20)
                
                # Progress bar settings
                total_tasks = len(data)
                progress_bar["maximum"] = total_tasks
                progress_bar["value"] = 0

                # Create a label for status updates
                status_label = Label(popup, text="Processing...")
                status_label.pack()

                # Update the popup to ensure it's drawn
                popup.update()

                # create objects to store the data
                X_calls = []
                Y_calls = []

                # filter the dataframe to the call type you want (at the moment it is only setup to extract tweeps)
                data = data[data['call_type'] == type] # keep only the specified calls  
                
                # carry out extraction on each row in the filtered dataframe
                for i, (index, row) in enumerate(data.iterrows(), 1):
                    # Update progress bar and label
                    progress_bar["value"] = i
                    status_label.config(text=f"Processing {i}/{total_tasks}")
                    popup.update()
                    file_to_find = os.path.join(folder_path, row['sound.files'])
                    
                    if os.path.exists(file_to_find):    
                                    x, y = self.get_tweep(row['sound.files'], folder_path, row['start'], row['ID'], row['Selection'])
                                    # we only want to store the array if it contains information
                                    if x is None:
                                        log_message(f"Skipped call {row['ID']}_{row['Selection']}")
                                        continue
                                    if x.size > 0:
                                        X_calls.append(x)
                                        Y_calls.append(y) 
                                                                               
                    else:
                        print('Cannot find', folder_path + row['sound.files'])  
      
                # save the unfiltered array 
                self.save_to_pickle_get_calls(file_name, X_calls, 'tweep_unfiltered.pkl', Y_calls, 'tweep_unfiltered_labels.pkl')
                print('Saving images...')

                # save pngs to file so you can do the filtering afterwards
                self.save_images(X_calls, Y_calls, file_name)

                # Cleanup after processing is done
                status_label.config(text="Done!")
                progress_bar["value"] = total_tasks
                popup.update()

                # Optionally, close the popup automatically after a delay
                time.sleep(2)
                popup.destroy()

            # Define a function to set the data file
    def final_combo(self):
                global data_file
                data_file = self.open_csv_file(self.data_file_path.get())
                print('data_file:\n ', data_file, '\nfolder_path: ', folder_path, '\ncall_type: ', self.selected_code.get())
                self.extract_tweeps(data_file, self.audiopath_var.get(), self.selected_code.get(), os.path.splitext(os.path.basename(self.meta_data_var.get()))[0])

    def button_8_click_get_calls(self):
                folder_path = self.audiopath_var.get()
                code = self.selected_code.get()
                meta_data = self.meta_data_var.get()
                
                if code != 'Ch':
                    messagebox.showwarning("Warning", "Only Churr calls currently supported")
                    return

                if not folder_path or not code or not meta_data:
                    messagebox.showwarning("Warning", "Please fill in all required fields!")
                    return

                if not isinstance(folder_path, str) or not isinstance(code, str) or not isinstance(meta_data, str):
                    messagebox.showwarning("Warning", "Please fill in all required fields with valid values!")
                    return

                if not folder_path.strip() or not code.strip() or not meta_data.strip():
                    messagebox.showwarning("Warning", "Please fill in all required fields with non-empty values!")
                    return

                print("button_8 clicked")
                self.final_combo()  