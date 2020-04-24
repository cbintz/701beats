import librosa
import librosa.display
import IPython.display as ipd
from ipywidgets import widgets
import matplotlib.pyplot as plt
import glob
import numpy as np
from array import *

audio_files = glob.glob('./audio/*.wav')

print(len(audio_files))

b = list()

#Song Data: B[i][0]
#Tempo: B[i][1]
#Chroma: B[i][2]
#MFCC: B[i][3]

sr = 44100
for i in range(0,len(audio_files)):

    #create list to store song data
    song_data = list()
    b.append(song_data)

    #create unique song tag
    song = 'song' + str(i)

    #load the song data
    song, sr = librosa.load(audio_files[i], offset = 10, duration = 30)

    #add the song to the song data list
    b[i].append(song)

    #extract beat and add to feature list
    onset_env = librosa.onset.onset_strength(b[i][0], sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    b[i].append(tempo)

    #extract chroma feature and add to feature list
    chroma_cq = librosa.feature.chroma_cqt(b[i][0], sr=sr)
    b[i].append(chroma_cq)

    #extract mfccs and add to feature list
    mfcc = librosa.feature.mfcc(b[i][0], sr=sr)
    b[i].append(mfcc)


#plot chroma_cq
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(b[0][2], y_axis='chroma')
plt.title('chroma_cq1')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(b[1][2], y_axis='chroma', x_axis='time')
plt.title('chroma_cq2')
plt.colorbar()
plt.tight_layout()
plt.show()

#plot mfcc
plt.figure(figsize=(10, 4))
librosa.display.specshow(b[0][3], x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

#x1 , sr = librosa.load(audio_files[0], sr = 44100, offset = 10, duration = 30)
#x2 , sr = librosa.load(audio_files[1], sr = 44100, offset = 10, duration = 30)
#x3 , sr = librosa.load(audio_files[2], sr = 44100, offset = 10, duration = 30)

#chroma_cens1 = librosa.feature.chroma_cens(x1, sr=sr)
