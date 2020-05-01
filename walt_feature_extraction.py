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

b = list()
mfcc_originals = list() #for plotting
chroma_originals = list() #for plotting


#Song filepath: B[i][0]
#Raw Song: B[i][1]
#Song_harmonic: B[i][2]
#Song_percussive: B[i][3]
#Tempo: B[i][4]
#Chromagram: B[i][5]
#mfcc1: B[i][6]
#beat features: B[i][7]
#beat chroma: B[i][8]
#chroma_cq: B[i][9]
#mfcc2: B[i][10]


fs = 44100
for i in range(0,len(audio_files)):
    #create list to store song data
    song_data = list()
    b.append(song_data)
    b[i].append(audio_files[i])

    #create unique song tag
    song = 'song' + str(i)

    #load the song data
    song, sr = librosa.load(audio_files[i], duration = 30)

    #set hop length
    hop_length = 512

    #separate song into harmonic and percussive components
    song_harmonic, song_percussive = librosa.effects.hpss(song)

    #beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=song_percussive,
                                                 sr=sr)
    tempo = [tempo]

    # compute mfcc features from the raw signal
    mfcc1 = librosa.feature.mfcc(y=song, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc1)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc1, mfcc_delta]),
                                        beat_frames)


    #calculate a chromagram from the harmonic component
    chromagram = librosa.feature.chroma_cqt(y=song_harmonic,
                                            sr=sr)

    #synchronize chroma and beat frames (beat events)
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

    #add the song to the song data list
    b[i].append(song)
    b[i].append(song_harmonic)
    b[i].append(song_percussive)
    b[i].append(tempo)
    b[i].append(chromagram)
    b[i].append(mfcc1)
    b[i].append(beat_features)
    b[i].append(beat_chroma)

    #extract chroma feature and add to feature list
    chroma_cq_og = np.array(librosa.feature.chroma_cqt(b[i][1], sr=sr))
    chroma_originals.append(chroma_cq_og)
    chroma_cq = chroma_cq_og.reshape(15504)
    b[i].append(chroma_cq)

    #extract mfccs and add to feature list
    mfcc_og = np.array(librosa.feature.mfcc(b[i][1], sr=sr))
    mfcc_originals.append(mfcc_og)
    # feature scaling made clusters more inaccurate, commenting out for now
    # scaler = MinMaxScaler(feature_range = (-1, 1))
    # scaler.fit(mfcc)
    # mfcc_scale = scaler.transform(mfcc)
    # mfcc_scale = mfcc_scale.reshape(25840)
    mfcc2 = mfcc_og.reshape(25840)
    # b[i].append(mfcc_scale)
    b[i].append(mfcc2)

mfcc_total = []
for i in range(len(b)):
    mfcc_total.append(b[i][10])

tempo_total = []
for i in range(len(b)):
    tempo_total.append(b[i][4])
tempo_total = np.array(tempo_total)


chroma_total = []
for i in range(len(b)):
    chroma_total.append(b[i][9])

#kmeans with mfcc
kmeans = KMeans(n_clusters=6, max_iter=100).fit(mfcc_total)
for j in range(6):
    print("mfcc cluster " + str(j+1) + ": ", end = "")
    for i in range(len(np.where(kmeans.labels_ == j)[0])):
        song_index = np.where(kmeans.labels_ == j)[0][i]
        song_path = audio_files[song_index]
        song = song_path[8:-4]
        print(song, end = ", ")
    print("", end = "\n")


#kmeans with tempo
kmeans2 = KMeans(n_clusters=6, max_iter=100).fit(tempo_total)
for j in range(6):
    print("tempo cluster " + str(j+1) + ": ", end = "")
    for i in range(len(np.where(kmeans2.labels_ == j)[0])):
        song_index = np.where(kmeans2.labels_ == j)[0][i]
        song_path = audio_files[song_index]
        song = song_path[8:-4]
        print(song, end = ", ")
    print("", end ="\n")



#kmeans with chroma
kmeans3 = KMeans(n_clusters=6, max_iter=100).fit(chroma_total)
for j in range(6):
    print("chroma cluster " + str(j+1) + ": ", end = "")
    for i in range(len(np.where(kmeans3.labels_ == j)[0])):
        song_index = np.where(kmeans3.labels_ == j)[0][i]
        song_path = audio_files[song_index]
        song = song_path[8:-4]
        print(song, end = ", ")
    print("", end ="\n")


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
