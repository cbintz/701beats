import librosa
import librosa.display
import IPython.display as ipd
from ipywidgets import widgets
import matplotlib.pyplot as plt
import glob
import numpy as np
from array import *
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

audio_files = glob.glob('./audio/*.wav')

# 0, 3, 8 = rap
# 1, 4, 6 = edm
# 2, 5, 7 = pop

b = list()
mfcc_originals = list() #for plotting
chroma_originals = list() #for plotting

#Song Data: B[i][0]
#Tempo: B[i][1]
#Chroma: B[i][2]
#MFCC: B[i][3]

fs = 44100
for i in range(0,len(audio_files)):

    #create list to store song data
    song_data = list()
    b.append(song_data)

    #create unique song tag
    song = 'song' + str(i)

    #load the song data
    song, sr = librosa.load(audio_files[i], duration = 30)

    #add the song to the song data list
    b[i].append(song)

    #extract beat and add to feature list
    onset_env = librosa.onset.onset_strength(b[i][0], sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    b[i].append(tempo)

    #extract chroma feature and add to feature list
    chroma_cq_og = np.array(librosa.feature.chroma_cqt(b[i][0], sr=sr))
    chroma_originals.append(chroma_cq_og)
    chroma_cq = chroma_cq_og.reshape(15504)
    b[i].append(chroma_cq)

    #extract mfccs and add to feature list
    mfcc_og = np.array(librosa.feature.mfcc(b[i][0], sr=sr))
    mfcc_originals.append(mfcc_og)
    # feature scaling made clusters more inaccurate, commenting out for now
    # scaler = MinMaxScaler(feature_range = (-1, 1))
    # scaler.fit(mfcc)
    # mfcc_scale = scaler.transform(mfcc)
    # mfcc_scale = mfcc_scale.reshape(25840)
    mfcc = mfcc_og.reshape(25840)
    # b[i].append(mfcc_scale)
    b[i].append(mfcc)

mfcc_total = []
for i in range(len(b)):
    mfcc_total.append(b[i][3])

tempo_total = []
for i in range(len(b)):
    tempo_total.append(b[i][1])
tempo_total = np.array(tempo_total)

chroma_total = []
for i in range(len(b)):
    chroma_total.append(b[i][2])

#kmeans with mfcc
kmeans = KMeans(n_clusters=3, max_iter=100).fit(mfcc_total)
print("mfcc cluster 1:", np.where(kmeans.labels_ == 0)[0]) #shows song indices in cluster 0
print("mfcc cluster 2:", np.where(kmeans.labels_ == 1)[0]) #shows song indices in cluster 1
print("mfcc cluster 3:", np.where(kmeans.labels_ == 2)[0]) #shows song indices in cluster 2


#kmeans with tempo
kmeans2 = KMeans(n_clusters=3, max_iter=100).fit(tempo_total)
print("tempo cluster 1:", np.where(kmeans2.labels_ == 0)[0]) #shows song indices in cluster 0
print("tempo cluster 2:", np.where(kmeans2.labels_ == 1)[0]) #shows song indices in cluster 1
print("tempo cluster 3:", np.where(kmeans2.labels_ == 2)[0]) #shows song indices in cluster 2

#kmeans with chroma
kmeans3 = KMeans(n_clusters=3, max_iter=100).fit(chroma_total)
print("chroma cluster 1:", np.where(kmeans3.labels_ == 0)[0]) #shows song indices in cluster 0
print("chroma cluster 2:", np.where(kmeans3.labels_ == 1)[0]) #shows song indices in cluster 1
print("chroma cluster 3:", np.where(kmeans3.labels_ == 2)[0]) #shows song indices in cluster 2



#feature scaling: didn't really help and made weird arrays, but keeping here in case we want to revisit

#print(b[0][3])
# mfccs = b[0][3]
# mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
# # print (mfccs.mean(axis=1))
# # print (mfccs.var(axis=1))
# # print(mfccs)
# song, sr = librosa.load(audio_files[0], offset = 10, duration = 30)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()

# song, sr = librosa.load(audio_files[i])
# librosa.onset.onset_detect(song, sr=sr)
# tempo, beats = librosa.beat.beat_track(song, sr=sr)
# beat_times = librosa.frames_to_time(beats, sr=sr)
# onset_samples = librosa.frames_to_samples(beats)
#
# def extract_features(x, fs):
#     feature_1 = librosa.feature.mfcc(x, sr=fs)
#     return [feature_1]
#
# frame_sz = int(sr*0.100)
# # for i in range(len(onset_samples)):
# #     onset_samples[i] = int(onset_samples[i])
# features = np.array([extract_features(song[i:i+frame_sz], sr) for i in onset_samples])
# print(features.shape)
# features = features.reshape(59,100)
#
#
# #print(features)
# scaler = MinMaxScaler(feature_range = (-1, 1))
# scaler.fit(features)
# print(scaler.transform(features))

#plot chroma_cq
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(chroma_originals[0], y_axis='chroma')
plt.title('chroma_cq1')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(chroma_originals[1], y_axis='chroma', x_axis='time')
plt.title('chroma_cq2')
plt.colorbar()
plt.tight_layout()
plt.show()

#plot mfcc
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_originals[0], x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
