from  pydub import AudioSegment
from pydub.playback import play

import os
import sys
import librosa
import librosa.display
import librosa.core
import math
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matplotlib.pyplot import specgram
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.io.wavfile import write

from matplotlib.ticker import ScalarFormatter

import threading
import time
from datetime import timedelta

import utils_video

# Set the Matplotlib backend to TkAgg
plt.switch_backend('TkAgg')





def animate_time_domain_plot(audio_from_one_mic, fs, path_to_music_file, play_audio = False, play_video = False, video_filename = None):

    def play_music(path):
        # Load and play music
        song = AudioSegment.from_wav(path)
        play(song)


    if play_audio:
        # Play music from path to music file in a separate thread
        play_thread = threading.Thread(target=play_music, args=(path_to_music_file,))
        play_thread.start()

    if play_video and video_filename is not None:
        # Play video in a separate thread
        video_thread = threading.Thread(target=utils_video.play_video, args=(video_filename,))
        video_thread.start()

    # Create a time array based on the audio signal length and sampling rate
    duration = len(audio_from_one_mic) / fs
    time = np.linspace(0, duration, len(audio_from_one_mic))
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(time, audio_from_one_mic)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    # Initialize the vertical line
    vertical_line = ax.axvline(x=time[0], color='r', linestyle='--')
    
    # Function to update the vertical line position
    def update(frame):

        if frame < 29:
            # Update the position of the vertical line
            time_index = int(fs*(interval_msec/1000)*frame)
            # print(f"frame: {frame}, time {time[time_index]+int(math.ceil(interval_msec/1000))}")
            vertical_line.set_xdata(time[time_index]+int(math.ceil(interval_msec/1000)) )
        return vertical_line,
    
    # Convert the interval from milliseconds to seconds (divide by 1000)
    interval_msec = 250

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(time), interval= interval_msec, blit=True)

    plt.show()

    # Close the plot window when the animation finishes
    plt.close(fig)

    if play_video and video_filename is not None:
        # Wait for the video to finish playing and close the video window
        video_thread.join()
        

#return the index and time of when the first motor sound is detected
def detect_motor_sound_trigger(audio_from_one_mic, fs, threshold=0.2):
    #input: audio_from_one_mic
    #output: index, time

    # Compute the amplitude envelope (just a simple absolute value)
    amplitude_envelope = np.abs(audio_from_one_mic)

    # Find where the amplitude exceeds the threshold
    change_points = np.where(amplitude_envelope > threshold)[0]

    #convert change point indexes to actual time
    change_points_time = []
    for index in change_points:
        change_points_time.append(index/fs)

    # print(f"change_points: {change_points}, change_points_time: {change_points_time}")

    #return the 0th index of the change_points_time

    return change_points[0], change_points_time[0]









def load_wav_to_npy(path_directory, input_file_list, save_directory):
# data loader 
# input: path directory, good list, bad list
# output: good_audio_list, bad_audio_list
# Dim is len of list [ [5 mic x Time], ..., [5 mic x Time] ]  

    num_mic = 5


    all_trial_all_mic = []

    #iterate through a given file_list
    for trial_name in (input_file_list):
        single_trial_all_mic = []

        # for 5 mics 
        for mic_index in range(1,num_mic+1):
            #TODO: load string trial_name using path_directory with trial_name and varying mic_index
            filename_ = os.path.join(path_directory, f"{trial_name}_{mic_index}.wav")
#             print(f"filename: {filename_}")

            #TODO: load wav file "t1_m1.wav through t1_m5.wav"
            mic_loaded_, fs = librosa.load(filename_, mono=False, sr=44100)

            single_trial_all_mic.append(mic_loaded_)

        #save
        single_trial_all_mic_np = np.array(single_trial_all_mic)
        save_filename = os.path.join(save_directory, f"t{trial_name}.npy")
        np.save(save_filename, single_trial_all_mic_np)

        #TODO: append the single_trial into trial_list
        all_trial_all_mic.append(single_trial_all_mic)

    return all_trial_all_mic,fs


# load t1~25.npy files and concatenate them into Trial x 5 x Time
def load_npy_files(path_directory, input_file_list):

    total_trials_all_mic = []

    #TODO: iterate through a given file_list
    for item in input_file_list:
        file_name = os.path.join(path_directory, f"t{item}.npy")
        trial_all_mic_audio = np.load(file_name, allow_pickle=True)
        #TODO: append the single_trial into trial_list
        total_trials_all_mic.append(trial_all_mic_audio)

    return total_trials_all_mic


# Function to animate spectrogram with vertical bar indicating the current time with the given audio 
def animate_spectrogram(audio, fs, save_animation_flag, save_filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle('Spectrogram Animation (Mic 1)', y=1.0, fontsize=10)

    # Create the spectrogram
    D = librosa.stft(audio, hop_length=512, n_fft=2048)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # print(f"size of S_db: {S_db.shape}")
    audio_len_msec = len(audio)/fs*1000
    S_db_col = S_db.shape[1]
    S_db_row = S_db.shape[0]
    im = ax.imshow(S_db, origin='lower', aspect='auto', cmap='viridis', extent=[0, len(audio) / fs, 0, fs / 2])


    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.set_label('Intensity (dB)', labelpad=10)

    # Hide grid lines
    ax.grid(False)

    # Set y-axis scale to log
    freq_ticks = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # Logarithmic frequency ticks
    ax.set_yscale('log')
    ax.set_yticks(freq_ticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())  # Use the correct formatter
    ax.set_ylim([freq_ticks[0], freq_ticks[-1]])  # Set y-axis limits

    # Add vertical line to indicate current time
    line = ax.axvline(0, color='r')

    # print(f"audio_len_msec/S_db_col {audio_len_msec/S_db_col}, where audio_len_msec {audio_len_msec} and S_db_col {S_db_col}")

    # Function to update the vertical line
    def update_line(frame):
        # global music_thread_started
        

        # if not music_thread_started:
            # music_thread.start()
            # music_thread_started = True

        fine_tuned_time_param = 1
        current_time = frame * audio_len_msec / S_db_col/ 1000 *fine_tuned_time_param
        print(f"current_time: {current_time}, frame : {frame}/{S_db_col}")
        
        # Update the vertical line to the current time
        line.set_xdata(current_time)
            

        return line,

    # Animate the vertical line
    ani = FuncAnimation(fig=fig, func=update_line, frames=S_db_col, interval=audio_len_msec/S_db_col,  repeat=False)
  
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Ensure proper spacing and accommodate colorbar
    
    if save_animation_flag:
        # Save the animation as a video file (e.g., .mp4)
        FFwriter = matplotlib.animation.FFMpegWriter(fps=10)
        ani.save(save_filename, writer = FFwriter)



    plt.show()
# ================================================================================================



# function to plot time domain and listen 
def plot_time_dmain_single_trial(single_trial_all_mic):
    fs = 44100
    audio_input1_trim = single_trial_all_mic[0]
    audio_input2_trim = single_trial_all_mic[1]
    audio_input3_trim = single_trial_all_mic[2]
    audio_input4_trim = single_trial_all_mic[3]
    audio_input5_trim = single_trial_all_mic[4]

    time_axis = np.linspace(0, len(audio_input1_trim) / fs, len(audio_input1_trim))

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Audio channels Plot')

    ax1 = plt.subplot(3, 1, 1)
    librosa.display.waveplot(audio_input1_trim, sr=fs)
    plt.title("mic1")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylim([-1, 1])
    ax1.xaxis.set_label_text('')

    ax2 = plt.subplot(3, 1, 2)
    librosa.display.waveplot(audio_input2_trim, sr=fs)
    plt.title("mic2")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylim([-1, 1])
    ax2.xaxis.set_label_text('')

    ax3 = plt.subplot(3, 1, 3)
    librosa.display.waveplot(audio_input3_trim, sr=fs, x_axis='time')
    plt.title("mic3")
    ax3.set_ylim([-1, 1])
    ax3.set_xlabel("Time")  # Set x-axis label to "Time"

  
    
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()

    return fig

def plot_time_domain_all_mic(trimmed_trials_all_mic, trial_index):
 
    fs = 44100
    audio_input1_trim = trimmed_trials_all_mic[trial_index][0]
    audio_input2_trim = trimmed_trials_all_mic[trial_index][1]
    audio_input3_trim = trimmed_trials_all_mic[trial_index][2]
    audio_input4_trim = trimmed_trials_all_mic[trial_index][3]
    audio_input5_trim = trimmed_trials_all_mic[trial_index][4]
    
    time_axis = np.linspace(0, len(audio_input1_trim) / fs, len(audio_input1_trim))

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Audio channels Plot')

    ax1 = plt.subplot(3, 1, 1)
    librosa.display.waveplot(audio_input1_trim, sr=fs)
    plt.title("mic1")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylim([-1, 1])
    ax1.xaxis.set_label_text('')

    ax2 = plt.subplot(3, 1, 2)
    librosa.display.waveplot(audio_input2_trim, sr=fs)
    plt.title("mic2")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylim([-1, 1])
    ax2.xaxis.set_label_text('')

    ax3 = plt.subplot(3, 1, 3)
    librosa.display.waveplot(audio_input3_trim, sr=fs, x_axis='time')
    plt.title("mic3")
    ax3.set_ylim([-1, 1])
    ax3.set_xlabel("Time")  # Set x-axis label to "Time"

  
    
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()

    return fig

# Function to plot time domain for mic1_trim of all trials in a 6x4 grid
def plot_time_domain_all_trial(trimmed_trials_all_mic, fs):
    num_trials = len(trimmed_trials_all_mic)

    fig, axes = plt.subplots(6, 4, figsize=(18, 18))
    fig.suptitle('Audio Channels Plot for All Trials (Mic 1)', y=1.02)

    for row in range(6):
        for col in range(4):
            trial_index = row * 4 + col
            if trial_index < num_trials:
                mic1_trim = trimmed_trials_all_mic[trial_index][0]

                time_axis = np.linspace(0, len(mic1_trim) / fs, len(mic1_trim))

                axes[row, col].plot(time_axis, mic1_trim)
                axes[row, col].set_title(f"Trial {[trial_index+1]}")
                axes[row, col].set_ylim([-1, 1])
                axes[row, col].set_xlabel("Time")  # Add x-axis label

                #add red vertical bar at time 5.7 second
                axes[row, col].axvline(x=6.2, color='r', linestyle='--')


            else:
                axes[row, col].axis('off')  # Turn off empty subplots

    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()

def save_wav_from_npy(input_npy, save_file):
    librosa.output.write_wav(save_file, input_npy, sr=44100)

if __name__ == "__main__":
    dataset_path = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/"
    save_directory = "/home/marklee/IAM/vibration_project/ISU_audio_07_2023/dataset_np/"

    # load wav files and save as npy files
    # total_list = numbers_list = [i for i in range(1, 26)]
    # total_trial_all_mic,fs = load_wav_to_npy(dataset_path, total_list, save_directory)


    # ------------------ animate spectrogram ------------------------------------------
    # fs = 44100
    # music_thread_started = False

    # song_data = AudioSegment.from_wav("/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/1_1.wav")

    # file_name = os.path.join(save_directory, f"t1.npy")
    # trial_all_mic_audio = np.load(file_name, allow_pickle=True)

    # audio_data = trial_all_mic_audio[0]
    # print(f"length of audio {len(audio_data)/fs}")

    # # Setup a separate thread to play the music
    # # music_thread = threading.Thread(target=play, args=(song_data,))
    # animate_spectrogram(audio_data, fs)



    # ----------------------- trim audio npy files -------------------------------------
    #==================================================================================================
    # load_directory = "/home/marklee/IAM/vibration_project/ISU_audio_07_2023/dataset_np/full_npy/"
    # save_directory = "/home/marklee/IAM/vibration_project/ISU_audio_07_2023/dataset_np/"

    # fs = 44100
    # total_list = numbers_list = [i for i in range(1, 26)]
    # total_trial_all_mic = load_npy_files(load_directory, total_list)

    # # print(f"total_trial_all_mic_np shape: {len(total_trial_all_mic)}")
    # # print(f"len of audio {len(total_trial_all_mic[0][0])/fs}")

    

    # # ----------------------------- MODIFY HERE -----------------------------
    # trial = 25
    # # plot_time_domain_all_mic(total_trial_all_mic, trial-1) #remember it's 0th index but 1.wav file for trials

    # # # trim audio
    
    # trigger_time = 2
    # #----------------------------------------------------------
    # before_trigger = 1
    # trigger_duration = 7
    # start_index = int(trigger_time-before_trigger)*fs
    # end_index = int(trigger_time+trigger_duration)*fs

    # #trim audio at trigger time
    # single_trial_all_mic = []
    # for i in range (5):
    #     single_trial_all_mic_np = total_trial_all_mic[trial-1][i][start_index:end_index]
    #     single_trial_all_mic.append(single_trial_all_mic_np)
    
    # single_trial_all_mic_np = np.array(single_trial_all_mic)

    # print(f"single_trial_all_mic_np shape: {len(single_trial_all_mic)}, seconds: {len(single_trial_all_mic[0])/fs}")

    # #save npy file
    # save_filename = os.path.join(save_directory, f"t{trial}.npy") #remember it's 0th index but 1.wav file for trials
    # np.save(save_filename, single_trial_all_mic_np)

    # ==================================================================================================
    