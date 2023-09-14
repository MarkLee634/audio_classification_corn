# Import everything needed to edit video clips 
from moviepy.editor import *

import os
import time
dataset_path = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/raw_video/"
save_path = "/rosbag/audio/ISU_audio_dataset/insertion/dataset_field/"

buffer_before = 1 #seconds
buffer_duration = 7

def trim_video(input_video, key_point_time):
    filename_ = os.path.join(dataset_path, f"{input_video}.MOV")
    clip = VideoFileClip(filename_) 

    #TODO: print the duration of the video
    video_duration = clip.duration
    print("Video Duration:", video_duration)

    #TODO: trim clip at key point time with buffer duration before and after
    start_time = max(0, key_point_time - buffer_before)
    end_time = min(video_duration, key_point_time + buffer_duration)
    
    trimmed_clip = clip.subclip(start_time, end_time)


    #TODO: save trimmed clip
    savename_ = os.path.join(save_path, f"{input_video}.MOV")
    trimmed_clip.write_videofile(savename_, codec="libx264")

    #usage
    #trim_video("6", 9.5)


def rename_video_files(source_directory, target_directory):

    start_number = 7
    end_number = 25

    for number in range(start_number, end_number + 1):
        old_filename = os.path.join(source_directory, f"IMG_{4472 + number - start_number}.MOV")
        new_filename = os.path.join(target_directory, f"{number}.MOV")
        
        print(f"Renaming {old_filename} to {new_filename}")
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
            print(f"Renamed {old_filename} to {new_filename}")
        else:
            print(f"File {old_filename} does not exist.")

    print("Renaming complete.")
    
    #usage
    # source_path = "/home/marklee/Downloads/drive-download-20230905T204033Z-001/"
    # save_path = "/home/marklee/Downloads/"
    # rename_video_files(source_path, save_path)


def play_video(video_path):
    width = 640
    height = 480
    video = VideoFileClip(video_path)
    video = video.resize((width, height))

    # Set the audio of the video to None (no sound)
    video = video.set_audio(None)
    
    video.preview()
    

# ===========================================================================================    

if __name__ == "__main__":
    trim_video("25", 5)   
