from moviepy.editor import *
import librosa
import face_recognition
from sklearn.cluster import KMeans

import numpy as np
import joblib
import os

hop_length = 512

### audio fragment
class audioFrag:
    def __init__(self, start, end, src, mfcc):
        self.pred = [start, end] # subclip duration
        self.src = src # video path
        self.mfcc = mfcc # mfcc extracted by librosa

def extractMFCC(audio):
    y, sr = librosa.load(audio, sr=None) # sr = 44100
    feature = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=24, hop_length=hop_length) # ceil( y.shape / hop_length )
    return feature

### save the list which stores all audioFrag(with features)
def extractFeatureFromVideo(video, interval, savePath): 
    features = []
    raw = VideoFileClip(video)
    t = np.arange(0, raw.duration, interval)
    for i in range(t.shape[0]-1):
        seg = raw.subclip(t[i], t[i+1])
        segAudio = seg.audio
        segAudio.write_audiofile("temp.wav")
        mfcc = extractMFCC("temp.wav")
        os.remove("temp.wav")

        features.append(audioFrag(t[i], t[i+1], video, mfcc))
        print(f"Finished {100 * (i+1)/(t.shape[0]-1): .4f}%")
        
    
    joblib.dump(features, savePath)


def m_extractFeatureFromVideo(video, interval, savePath, t_list):
    features = []
    raw = VideoFileClip(video)
    #t = np.arange(0, raw.duration, interval)
    for i in range(len(t_list)):
        if(t_list[i] + interval > raw.duration):
            continue
        seg = raw.subclip(t_list[i], t_list[i] + interval)
        segAudio = seg.audio
        segAudio.write_audiofile("temp.wav")
        mfcc = extractMFCC("temp.wav")
        os.remove("temp.wav")

        features.append(audioFrag(t_list[i], t_list[i] + interval, video, mfcc))
        print(f"Finished {100 * (i + 1) / len(t_list): .4f}%")

    joblib.dump(features, savePath)

def cv_kmeans(video, audio, intervals):
    feature = extractMFCC(audio)
    print(feature.shape)
    mv = VideoFileClip(video)

    cv_features = []
    cv_feature_id = []
    tmp = feature.shape[1]
    for i in range(0, tmp):

        mul = min(i / feature.shape[1], 0.999)
        frame = mv.get_frame(mv.duration * mul)[:, :, ::-1]
        # cv.imshow(" ", frame)
        # cv.waitKey(0)
        face_landmarks_list = face_recognition.face_landmarks(frame)
        # pil_image = Image.fromarray(frame)
        # d = ImageDraw.Draw(pil_image)
        print("found: ", len(face_landmarks_list))
        if (len(face_landmarks_list) != 1):
            continue
        for face_landmarks in face_landmarks_list:

            # Print the location of each facial feature in this image
            features = []
            for facial_feature in face_landmarks.keys():
                # print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
                features.append(face_landmarks[facial_feature])

            # Let's trace out each facial feature in the image with a line!
            # for facial_feature in []:
            #     d.line(face_landmarks[facial_feature], width=5)
            # d.line(features[7][0:6], width=5)
            # d.line(features[8][0:6], width=5)
            # print(features[7])
            # print(features[8])
            # print(len(features[7]))
            top_lip = [[features[7][j][0], features[7][j][1]] for j in range(len(features[7]))]
            bottom_lip = [[features[8][j][0], features[8][j][1]] for j in range(len(features[8]))]
            bottom_lip = bottom_lip[0:6][::-1] + bottom_lip[6:12][::-1]
            lengths = [top_lip[j][0] for j in range(12)]
            length = max(lengths) - min(lengths)  # mouth length
            # print(top_lip)
            # print(bottom_lip)
            top_lip = np.array(top_lip)
            bottom_lip = np.array(bottom_lip)

            # print(length)
            # print((top_lip.reshape(-1) - bottom_lip.reshape(-1))/length)
            cv_features.append((top_lip.reshape(-1) - bottom_lip.reshape(-1)) / length)  # normalize
            cv_feature_id.append(i)
        print(i)
        # pil_image.show()

    total_list = []
    for it in intervals:
        clusterN = int(mv.duration / it / 10)
        kmeans = KMeans(n_clusters=clusterN)
        kmeans.fit(cv_features)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        print(centers)
        print(labels)

        t_list = []
        for center in centers:
            dist = float("inf")
            idx = 0
            for i in range(len(cv_features)):
                tmpd = np.linalg.norm(center - cv_features[i])
                if (tmpd < dist):
                    dist = tmpd
                    idx = cv_feature_id[i]
            t_list.append(mv.duration * idx / feature.shape[1])
        print(it, t_list)
        total_list.append(t_list)
    return total_list

if __name__ == "__main__":
    video = "./rsc/trump.mp4"
    audio = "./rsc/trump.wav"
    is_using_cv = False  # will be useful if you have a very long video
    intervals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    t_lists = []
    if(is_using_cv):
        t_lists = cv_kmeans(video, audio, intervals)

    if not(is_using_cv):
        for it in intervals:
            savePath = "./features/interval_"+str(it)+".sav"
            if not os.path.exists(savePath):
                extractFeatureFromVideo(video, it, savePath)
    else:
        for i in range(len(intervals)):
            savePath = "./features/interval_"+str(intervals[i])+".sav"
            if not os.path.exists(savePath):
                m_extractFeatureFromVideo(video, intervals[i], savePath, t_lists[i])

    pass