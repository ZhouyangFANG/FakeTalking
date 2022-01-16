from moviepy.editor import *
import librosa

import numpy as np
import joblib
import os
import shutil
import math

from sklearn.cluster import KMeans

from util import *



def findClosest(mfcc, features):
    idx = -1
    dist = float("inf")
    for i in range(len(features)):
        temp = np.linalg.norm(mfcc.reshape(1, -1) - features[i].mfcc.reshape(1, -1))
        if temp < dist:
            dist = temp
            idx = i
    return features[idx], dist

def match(audio, curPtr, interval, features):
    seg = audio.subclip(curPtr, curPtr+interval)
    seg.write_audiofile("temp.wav")
    mfcc = extractMFCC("temp.wav")
    os.remove("temp.wav")
    matched, dist = findClosest(mfcc, features)

    return matched, dist

def concatVideos(sequence, savePath="stitch.mp4"):
    seg = sequence[0]
    src = seg.src
    all_list = []
    video = VideoFileClip(src).subclip(seg.pred[0], seg.pred[1])
    video = video.set_audio(AudioFileClip("./TMP/0.wav"))
    all_list.append(video)
    for i in range(1, len(sequence)):
        seg = sequence[i]
        src = seg.src
        temp = VideoFileClip(src).subclip(seg.pred[0], seg.pred[1])
        temp = temp.set_audio(AudioFileClip("./TMP/"+str(i)+".wav"))
        all_list.append(temp)
    video = concatenate_videoclips(all_list)
    video.write_videofile(savePath)

if __name__ == "__main__":
    ### load list stores features
    intervals = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    dstAudio = AudioFileClip("./rsc/obama.wav") # input audio
    
    curPtr = 0 # current time
    end = dstAudio.duration - max(intervals) # the length of the input

    sequence = []
    if os.path.exists("./TMP"):
        shutil.rmtree("./TMP")
    os.mkdir("./TMP")

    while curPtr < end:
        
        temp_dist = float("inf") # for decide step
        temp_step = 0
        temp_matched = None

        for i in range(len(intervals)):
            interval = intervals[i]
            features = joblib.load("./features/interval_"+str(interval)+".sav")
            matched, dist = match(dstAudio, curPtr, interval, features)

            dim = math.ceil(interval*44100 / hop_length) # 44100 is the sampling rate; hop_length is defined in the util.py

            dist = dist / math.sqrt(dim) # refine dist


            if dist < temp_dist:
                temp_dist = dist
                temp_step = interval
                temp_matched = matched
        # add dst audio
        temp_au = dstAudio.subclip(curPtr, curPtr + temp_step)
        temp_au.write_audiofile("./TMP/"+str(len(sequence))+".wav")

        curPtr += temp_step
        sequence.append(temp_matched)
        print(f"Finished {curPtr} / {end}")

    concatVideos(sequence)
    shutil.rmtree("./TMP")
    pass