from  pydub import AudioSegment
 
import os
import numpy as np

import utils_audio
import utils_video
import yaml

fs= 44100

def trim_raw_npy_files():
    #load full npy
    raw_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/raw_audio/"
    total_list = numbers_list = [i for i in range(1, 26)]

    #load saved npy files (indices from 1-25)
    total_trial_all_mic = utils_audio.load_npy_files(raw_load_directory, total_list)

    #plot time domain
    # utils_audio.plot_time_domain_all_trial(total_trial_all_mic,fs)

    trim_audio_to_motor()

def visualize_trimmed_files():
    #visualize all the trimmed audio
    sync_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/sync_audio/"
    total_list = numbers_list = [i for i in range(1, 26)]

    #load saved npy files (indices from 1-25)
    total_trial_all_mic = utils_audio.load_npy_files(sync_load_directory, total_list)


    utils_audio.plot_time_domain_all_trial(total_trial_all_mic,fs)


def convert_wav_to_npy():
    dataset_path = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/"
    save_directory = "/home/marklee/IAM/vibration_project/ISU_audio_07_2023/dataset_np/"
    
    total_list = numbers_list = [i for i in range(1, 26)]

    #1. convert wav to npy (indices from 1-25) and save
    utils_audio.load_wav_to_npy(dataset_path, total_list, save_directory)

    

def load_and_animate_spectrogram():

    load_directory = "/home/marklee/IAM/vibration_project/ISU_audio_07_2023/dataset_np/"

    total_list = numbers_list = [i for i in range(1, 26)]

    # #2. load saved npy files (indices from 1-25)
    total_trial_all_mic = utils_audio.load_npy_files(load_directory, total_list)

    trial = 1
    print(f"total_trial_all_mic_np shape: {len(total_trial_all_mic)}")
    print(f"len of audio {len(total_trial_all_mic[trial-1][0])/fs}")
    # utils_audio.plot_time_domain_all_mic(total_trial_all_mic,trial-1)

    #3. trim audio

    #4. animate spectrogram
    save_movie_path = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/animation/"
    save_filename = os.path.join(save_movie_path, f"ani{trial}.mp4") #remember it's 0th index but 1.wav file for trials
    utils_audio.animate_spectrogram(total_trial_all_mic[trial-1][0], fs, save_animation_flag=False, save_filename = save_filename)


    #5. 

    # movie_path = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/"
    # video_file = os.path.join(movie_path, f"{trial}.MOV") #remember it's 0th index but 1.wav file for trials
    # utils_video.play_video(video_file)


def trim_audio_to_motor():
    load_directory = '/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/raw_audio/'
    total_list = numbers_list = [i for i in range(1, 26)]

    #load saved npy files (indices from 1-25)
    total_trial_all_mic = utils_audio.load_npy_files(load_directory, total_list)

    for index,audio_for_trial_i in enumerate (total_trial_all_mic):

        #print out length before trim
        print(f"length of audio before trim: {len(audio_for_trial_i[0])/fs}")

        #detect motor sound trigger
        trigger_index, trigger_time = utils_audio.detect_motor_sound_trigger(audio_for_trial_i[0], fs, threshold=0.05)

        #trim audio 
        motor_duration_sec = 8#5.7
        motor_duration_index = int(motor_duration_sec*fs)

        trimmed_audio_all_mic = []
        for mic_n in range(0,5):
            trimmed_audio = audio_for_trial_i[mic_n][trigger_index:trigger_index+motor_duration_index]
            trimmed_audio_all_mic.append(trimmed_audio)
        trimmed_audio_all_mic_np = np.array(trimmed_audio_all_mic)

        print(f"length of audio after trim: {len(trimmed_audio_all_mic_np[0])/fs}")

        #save trimmed audio
        save_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/sync_audio/"
        save_filename = os.path.join(save_directory, f"t{index+1}.npy")
        np.save(save_filename, trimmed_audio_all_mic_np)



        #visualize correctness by plotting and printing

        utils_audio.plot_time_domain_all_mic(total_trial_all_mic, index)
        
        # utils_audio.animate_time_domain_plot(audio_for_trial_i[0], fs)
        print(f"index: {index}, trigger_index: {trigger_index}, trigger_time: {trigger_time}")

        utils_audio.plot_time_dmain_single_trial(trimmed_audio_all_mic)

# convert npy to wav from trimmed files
def convert_npy_to_wav():
    #visualize all the trimmed audio
    sync_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/sync_audio/"
    total_list = numbers_list = [i for i in range(1, 26)]

    #load saved npy files (indices from 1-25)
    total_trial_all_mic = utils_audio.load_npy_files(sync_load_directory, total_list)

    for index,audio_for_trial_i in enumerate (total_trial_all_mic):
        #save trimmed audio
        save_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/sync_audio/"
        save_filename = os.path.join(save_directory, f"t{index+1}.wav")
        print(f" saving to {save_filename}")
        utils_audio.save_wav_from_npy(audio_for_trial_i[0], save_filename)

def animate_timeplot_video():
    #visualize all the trimmed audio
    sync_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/sync_audio/"
    move_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/trimmed_video/"
    total_list = numbers_list = [i for i in range(1, 26)]

    #load saved npy files (indices from 1-25)
    total_trial_all_mic = utils_audio.load_npy_files(sync_load_directory, total_list)

    while True:
        try:
            # Wait for user input
            trial_input = input("Enter a trial number (q to quit): ")
            
            if trial_input.lower() == 'q':
                # Exit the loop if 'q' is entered
                break

            audio_filename = os.path.join(sync_load_directory, f"t{trial_input}.wav")
            video_filename = os.path.join(move_load_directory, f"{trial_input}.MOV")

            # Animate the time domain plot according to the keyboard input
            utils_audio.animate_time_domain_plot(total_trial_all_mic[int(trial_input) - 1][0], fs, audio_filename, play_audio=True, play_video=True, video_filename=video_filename)

        except KeyboardInterrupt:
            # Exit the loop if Ctrl+C is pressed
            break

if __name__ == "__main__":
    print(f" -------- start -------- ")

    #1.preprocessing audio to trim npy files and then save wav file to play
    # trim_raw_npy_files()
    # visualize_trimmed_files()
    # convert_npy_to_wav()

    #2. animate time domain plot with audio and video
    # animate_timeplot_video()
    # =====================================================================================

    # Animate spectrogram
    sync_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/sync_audio/"
    move_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/trimmed_video/"
    total_list = numbers_list = [i for i in range(1, 26)]
    #load saved npy files (indices from 1-25)
    total_trial_all_mic = utils_audio.load_npy_files(sync_load_directory, total_list)

    while True:
        try:
            # Wait for user input
            trial_input = input("Enter a trial number (q to quit): ")
            
            if trial_input.lower() == 'q':
                # Exit the loop if 'q' is entered
                break

            audio_filename = os.path.join(sync_load_directory, f"t{trial_input}.wav")
            video_filename = os.path.join(move_load_directory, f"{trial_input}.MOV")

            # Animate the time domain plot according to the keyboard input
            utils_audio.animate_spectrogram(total_trial_all_mic[int(trial_input) - 1][0], fs, audio_filename, play_audio=True, play_video=True, video_filename=video_filename)

        except KeyboardInterrupt:
            # Exit the loop if Ctrl+C is pressed
            break






    # =====================================================================================
    # LABELS AND TIME PLOT
    #visualize all the trimmed audio
    # sync_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/sync_audio/"
    # move_load_directory = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/trimmed_video/"
    # total_list = numbers_list = [i for i in range(1, 26)]

    # #load labels
    # label_filename = "/home/marklee/github/audio_classification_corn/labels.yaml"

    # # Load YAML data from a file
    # with open(label_filename, 'r') as file:
    #     labels = yaml.load(file, Loader=yaml.FullLoader)

    # print(f"labels {labels}")

    # #load saved npy files (indices from 1-25)
    # total_trial_all_mic = utils_audio.load_npy_files(sync_load_directory, total_list)
    # utils_audio.plot_time_domain_all_trial_withlabels(total_trial_all_mic,fs, labels)

    #3. take only the 

    
    
    # =====================================================================================


        