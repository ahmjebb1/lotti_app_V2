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
from FilterCallsPage import FilterCallsPage

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
            
            self.button_8 = Button(self,text = "Run",command=lambda: (print("button_8 clicked"), self.button_8_click_get_calls()))
            self.button_8.grid(row = 3, column=0, sticky= "nsew", padx=5, pady=5, columnspan= 3)            

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
                    return None

            # these are the functions to extract the tweep from a Lotti churr call  
    def get_tweep(self, call, folder_path, start, call_id, selec):
                                #AJ librosa.load parameters are as follows:
                                    #offset - start reading after this time in seconds
                                    #duration - only load up to this much audio in seconds
                                    #sr - target sampling rate
                                y, sr = librosa.load(folder_path + call, offset = start, duration = 0.6, sr = 48000)

                                # --- Simplified waveform plot debug (Step 1) ---
                                waveform_debug_path = os.path.join(os.getcwd(), "debug_waveform")
                                if not os.path.exists(waveform_debug_path):
                                    os.makedirs(waveform_debug_path)

                                sanitized_filename = re.sub(r'[<>:"/\\|?*]', '', call_id)

                                # Use the waveform `y` and sample rate `sr` from earlier in the pipeline
                                # Only process for the first file
                                if len(images) > 0:
                                    # Plot and save waveform
                                    plt.figure(figsize=(10, 3))
                                    plt.plot(y, color='gray')
                                    plt.title(f"Waveform before noise reduction: {sanitized_filename}")
                                    plt.xlabel("Samples")
                                    plt.ylabel("Amplitude")
                                    plt.tight_layout()

                                    waveform_plot_file = os.path.join(waveform_debug_path, f"{sanitized_filename}_waveform.png")
                                    plt.savefig(waveform_plot_file)
                                    plt.close()

                                ## Noisereduce
                                #AJ Switched n_fft from 512 to 768 (1.5 x original value)
                                #AJ debugging switched n_fft to 1024
                                # nr.reduce_noise reduces noise in bioacoustics using a method called spectral gating. The 
                                # parameters are as follows:
                                    # thresh_n_mult_nonstationary: Non-stationary Noise Reduction: Continuously updates the 
                                    # estimated noise threshold over time. Default argument is 1.
                                    # n_fft: int, optional length of the windowed signal after padding with zeros. 
                                    # The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``. 
                                    # The default value, ``n_fft=2048`` samples, corresponds to a physical 
                                    # duration of 93 milliseconds at a sample rate of 22050 Hz. 
                                    # This value is well adapted for music signals. However, in speech processing, 
                                    # the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. 
                                    # In any case, we recommend setting ``n_fft`` to a power of two for 
                                    # optimizing the speed of the fast Fourier transform (FFT) algorithm., by default 1024
                                y = nr.reduce_noise(y=y, sr=sr, thresh_n_mult_nonstationary = 12,  n_fft = 768)

                                # Compute the Mel spectrogram
                                #AJ Switched n_fft from 512 to 768 (1.5 x original value)
                                #AJ changed fmin 2000 to 5000)
                                #AJ debugging: AJ switched fmin back to 2000, n_fft to 1024 and hop_length to 256
                                #
                                    # computing a mel spectrogram using a time-series input
                                    # sr = sampling rate
                                    # hop_length: number of samples between successive frames.
                                    # center: If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length].
                                    # n_mels:number of Mel bands to generate, 
                                    # fmin: lowest frequency (in Hz) 
                                    # fmax: highest frequency (in Hz)
                                S = librosa.feature.melspectrogram(y = y, sr=sr, n_fft = 768, hop_length=16, n_mels=128, fmin=5000, fmax=11000)

                                # === Normalize using decibel scale ===
                                S_dB = librosa.amplitude_to_db(S, ref=np.max)

                                # Convert to magnitude spectrogram (dB)
                                #AJ
                                    #Convert an amplitude spectrogram to dB-scaled spectrogram
                                    # ref = If scalar, the amplitude abs(S) is scaled relative to ref: 20 * log10(S / ref).
                                    # Zeros in the output correspond to positions where S == ref.
                                S_dB = librosa.amplitude_to_db(S, ref=np.max)

                                # Normalize the array to the range [0, 1]
                                S_dB_norm = librosa.util.normalize(S_dB, axis=0)

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

                                if not selected_size:
                                        min_size = 50
                                else:
                                        min_size = int(selected_size)

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

                                # Add frequency contours
                                contour_number = self.contour_no_combobox.get()
                                if not contour_number:
                                        num_contours = 10
                                else:
                                        num_contours = int(contour_number)
                                
                                min_contour_level = self.contour_min_combobox.get()
                                if not min_contour_level:
                                        min_level = 0.4
                                else:
                                        min_level = float(min_contour_level)

                                contour_levels = np.linspace(S_dB_norm.min() + (S_dB_norm.max() - S_dB_norm.min()) * min_level, S_dB_norm.max(), num_contours)

                                # Create a single array that represents the contour plot
                                contour_array = np.zeros_like(cleaned_spectrogram)

                                for i in range(num_contours - 1):
                                    contour_array[(cleaned_spectrogram >= contour_levels[i]) & (cleaned_spectrogram <= contour_levels[i + 1])] = i

                                sums = np.sum(contour_array, axis = 0)

                                peaks, _ = find_peaks(sums) 

                                # ensure that there is no failure by creating a fail safe if no peaks are detected
                                if len(peaks) > 0:
                                        call_start = np.min(peaks[0]-50) # give a little bit of space before the call
                                        # AJ increased the amount of space from 15 to 50
                                else:
                                        call_start = 1

                                # get the time of the start
                                    # Convert frame counts to time (seconds).
                                start_time = librosa.frames_to_time(call_start, sr = sr, hop_length = 16)
                                #AJ increased this from 0.1 to + 0.11
                                end_times = start_time + 0.11
                                end_frames = librosa.time_to_frames(end_times, sr = sr, hop_length = 16)

                                # Select the section of the Mel spectrogram corresponding to x and x + 200ms (AJ changed not sure about new value)
                                S_dB_section = contour_array[: , call_start:end_frames]
                                
                                call_id = f'{call_id}_{selec}'

                                return S_dB_section, call_id
        
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
                                            
                            zero_rows = np.where(np.all(np.isclose(img, 0, atol=1e-6), axis=1))[0]
                            debug_file.write(f"Fully zero Mel bands (row indices): {zero_rows}\n")
                                           
                            # Count and log unique values
                            unique_vals, counts = np.unique(img, return_counts=True)
                            debug_file.write("Unique values and counts (top 10):\n")
                            for val, count in list(zip(unique_vals, counts))[:10]:
                                debug_file.write(f"  Value: {val:.4f}, Count: {count}\n")
                                             
                            # Simulated reference value for normalization (e.g., np.max before conversion)
                            debug_file.write(f"Simulated reference value (np.max): {np.max(img)}\n")
                                               
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