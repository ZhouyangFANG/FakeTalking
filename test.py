from moviepy.editor import *
import librosa
import face_recognition

import numpy as np
from sklearn.cluster import KMeans
import joblib
import os
import cv2 as cv
from PIL import Image, ImageDraw

from util import *

# audio = AudioFileClip("./rsc/gt.mp3").subclip(0,8.0)
# audio.write_audiofile("gt_.mp3")
video = VideoFileClip("./rsc/raw.mp4")
video = video.subclip(0, 10)
au = video.audio
video.write_videofile("./rsc/short.mp4")
au.write_audiofile("./rsc/short.wav")

# y, sr = librosa.load("test.mp3", sr=None) # sr = 44100
# feature = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=24)
# print(feature.shape)


# features = joblib.load("interval_0.4.sav")
# print(features[0])

feature = extractMFCC("./rsc/raw.wav")
print(feature.shape)
i = 0
mv = VideoFileClip("./rsc/raw.mp4")
clusterN = int(mv.duration / 0.2 / 10)
cv_features = []
cv_feature_id = []
tmp = feature.shape[1]
for i in range(0, tmp):

    mul = min(i / feature.shape[1], 0.999)
    frame = mv.get_frame(mv.duration* mul)[:, :, ::-1]
    # cv.imshow(" ", frame)
    # cv.waitKey(0)
    face_landmarks_list = face_recognition.face_landmarks(frame)
    pil_image = Image.fromarray(frame)
    d = ImageDraw.Draw(pil_image)
    print("found: ",len(face_landmarks_list))
    if(len(face_landmarks_list) != 1):
        continue
    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        features = []
        for facial_feature in face_landmarks.keys():
            #print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
            features.append(face_landmarks[facial_feature])

        # Let's trace out each facial feature in the image with a line!
        # for facial_feature in []:
        #     d.line(face_landmarks[facial_feature], width=5)
        d.line(features[7][0:6], width=5)
        d.line(features[8][0:6], width=5)
        # print(features[7])
        # print(features[8])
        #print(len(features[7]))
        top_lip = [[features[7][j][0], features[7][j][1]] for j in range(len(features[7]))]
        bottom_lip = [[features[8][j][0], features[8][j][1]] for j in range(len(features[8]))]
        bottom_lip = bottom_lip[0:6][::-1]+ bottom_lip[6:12][::-1]
        lengths = [top_lip[j][0] for j in range(12)]
        length = max(lengths) - min(lengths)  # mouth length
        #print(top_lip)
        #print(bottom_lip)
        top_lip = np.array(top_lip)
        bottom_lip = np.array(bottom_lip)

        #print(length)
        #print((top_lip.reshape(-1) - bottom_lip.reshape(-1))/length)
        cv_features.append((top_lip.reshape(-1) - bottom_lip.reshape(-1))/length)  # normalize
        cv_feature_id.append(i)
    print(i)
    # pil_image.show()
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
        if(tmpd < dist):
            dist = tmpd
            idx = cv_feature_id[i]
    t_list.append(mv.duration * idx / feature.shape[1])
print(t_list)
    # pil_image.show()