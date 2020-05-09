import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

pop1 = '/Users/corinnebintz/Desktop/701/701beats/pop/Mistakes.m4a'
pop2 = '/Users/corinnebintz/Desktop/701/701beats/pop/Dont_Start_Now.m4a'
pop3 = '/Users/corinnebintz/Desktop/701/701beats/pop/Light_On.m4a'
x1 , sr = librosa.load(pop1, sr = 44100, offset = 10, duration = 30)
x2 , sr = librosa.load(pop2, sr = 44100, offset = 10, duration = 30)
x3 , sr = librosa.load(pop3, sr = 44100, offset = 10, duration = 30)


rap1 = '/Users/corinnebintz/Desktop/701/701beats/rap/SUGAR.m4a'
rap2 = '/Users/corinnebintz/Desktop/701/701beats/rap/3005.m4a'
rap3 = '/Users/corinnebintz/Desktop/701/701beats/rap/HUMBLE.m4a'

x4 , sr = librosa.load(rap1, sr = 44100, offset = 10, duration = 30)
x5 , sr = librosa.load(rap2, sr = 44100, offset = 10, duration = 30)
x6 , sr = librosa.load(rap3, sr = 44100, offset = 10, duration = 30)

#feature extraction

#BPMs
onset_env1 = librosa.onset.onset_strength(x1, sr=sr)
tempo1 = librosa.beat.tempo(onset_envelope=onset_env1, sr=sr)

onset_env2 = librosa.onset.onset_strength(x2, sr=sr)
tempo2 = librosa.beat.tempo(onset_envelope=onset_env2, sr=sr)

onset_env3 = librosa.onset.onset_strength(x3, sr=sr)
tempo3 = librosa.beat.tempo(onset_envelope=onset_env3, sr=sr)

onset_env4 = librosa.onset.onset_strength(x4, sr=sr)
tempo4 = librosa.beat.tempo(onset_envelope=onset_env4, sr=sr)

onset_env5 = librosa.onset.onset_strength(x5, sr=sr)
tempo5 = librosa.beat.tempo(onset_envelope=onset_env5, sr=sr)

onset_env6 = librosa.onset.onset_strength(x5, sr=sr)
tempo6 = librosa.beat.tempo(onset_envelope=onset_env6, sr=sr)

print(tempo1, tempo2, tempo3, tempo4, tempo5, tempo6)


#chromagrams
chroma_cens1 = librosa.feature.chroma_cens(x1, sr=sr)
chroma_cq1 = librosa.feature.chroma_cqt(x1, sr=sr)
chroma1 = scale(chroma_cq1, axis=1)

chroma_cens2 = librosa.feature.chroma_cens(x2, sr=sr)
chroma_cq2 = librosa.feature.chroma_cqt(x2, sr=sr)
chroma2 = scale(chroma_cq2, axis=1)

chroma_cens3 = librosa.feature.chroma_cens(x3, sr=sr)
chroma_cq3 = librosa.feature.chroma_cqt(x3, sr=sr)
chroma3 = scale(chroma_cq3, axis=1)

chroma_cens4 = librosa.feature.chroma_cens(x4, sr=sr)
chroma_cq4 = librosa.feature.chroma_cqt(x1, sr=sr)
chroma4 = scale(chroma_cq4, axis=1)

chroma_cens5 = librosa.feature.chroma_cens(x5, sr=sr)
chroma_cq5 = librosa.feature.chroma_cqt(x5, sr=sr)
chroma5 = scale(chroma_cq5, axis=1)

chroma_cens6 = librosa.feature.chroma_cens(x6, sr=sr)
chroma_cq6 = librosa.feature.chroma_cqt(x6, sr=sr)
chroma6 = scale(chroma_cq6, axis=1)

chroma1_linear = [i for sublist in chroma1 for i in sublist]
chroma2_linear = [i for sublist in chroma2 for i in sublist]
chroma3_linear = [i for sublist in chroma3 for i in sublist]
chroma4_linear = [i for sublist in chroma4 for i in sublist]
chroma5_linear = [i for sublist in chroma5 for i in sublist]
chroma6_linear = [i for sublist in chroma6 for i in sublist]

plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(chroma_cq1, y_axis='chroma')
plt.title('chroma_cq')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(chroma_cens1, y_axis='chroma', x_axis='time')
plt.title('chroma_cens')
plt.colorbar()
plt.tight_layout()
plt.show()

#mfccs
mfcc1 = librosa.feature.mfcc(x1, sr=sr)
mfcc1_delta = librosa.feature.delta(mfcc1, mode='nearest')
mfcc1_complete = np.concatenate((mfcc1, mfcc1_delta), axis=0)
mfcc1_scale = scale(mfcc1_complete, axis=1)

mfcc2 = librosa.feature.mfcc(x2, sr=sr)
mfcc2_delta = librosa.feature.delta(mfcc2, mode='nearest')
mfcc2_complete = np.concatenate((mfcc2, mfcc2_delta), axis=0)
mfcc2_scale = scale(mfcc2_complete, axis=1)

mfcc3 = librosa.feature.mfcc(x3, sr=sr)
mfcc3_delta = librosa.feature.delta(mfcc3, mode='nearest')
mfcc3_complete = np.concatenate((mfcc3, mfcc3_delta), axis=0)
mfcc3_scale = scale(mfcc3_complete, axis=1)

mfcc4 = librosa.feature.mfcc(x4, sr=sr)
mfcc4_delta = librosa.feature.delta(mfcc4, mode='nearest')
mfcc4_complete = np.concatenate((mfcc4, mfcc4_delta), axis=0)
mfcc4_scale = scale(mfcc4_complete, axis=1)

mfcc5 = librosa.feature.mfcc(x5, sr=sr)
mfcc5_delta = librosa.feature.delta(mfcc5, mode='nearest')
mfcc5_complete = np.concatenate((mfcc5, mfcc5_delta), axis=0)
mfcc5_scale = scale(mfcc5_complete, axis=1)

mfcc6 = librosa.feature.mfcc(x6, sr=sr)
mfcc6_delta = librosa.feature.delta(mfcc6, mode='nearest')
mfcc6_complete = np.concatenate((mfcc6, mfcc6_delta), axis=0)
mfcc6_scale = scale(mfcc6_complete, axis=1)



#https://github.com/smarsland/AviaNZ/blob/master/Clustering.py
mfcc1_linear = [i for sublist in mfcc1_scale for i in sublist]


mfcc2_linear = [i for sublist in mfcc2_scale for i in sublist]


mfcc3_linear = [i for sublist in mfcc3_scale for i in sublist]

mfcc4_linear = [i for sublist in mfcc4_scale for i in sublist]

mfcc5_linear = [i for sublist in mfcc5_scale for i in sublist]

mfcc6_linear = [i for sublist in mfcc6_scale for i in sublist]


plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc1, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

#KMEANS STEP

chroma = [np.array(chroma1_linear), np.array(chroma2_linear), np.array(chroma3_linear), np.array(chroma4_linear), np.array(chroma5_linear), np.array(chroma6_linear)]
mfcc = [np.array(mfcc1_linear), np.array(mfcc2_linear), np.array(mfcc3_linear), np.array(mfcc4_linear), np.array(mfcc5_linear), np.array(mfcc6_linear)]

tempo = np.array([tempo1, tempo2, tempo3, tempo4, tempo5, tempo6])

kmeans = KMeans(n_clusters=2, max_iter=100).fit(mfcc)
print([kmeans.labels_, kmeans.cluster_centers_])
print(np.where(kmeans.labels_ == 0)[0])
print(np.where(kmeans.labels_ == 1)[0])

kmeans = KMeans(n_clusters=2, max_iter=100).fit(tempo)
print([kmeans.labels_, kmeans.cluster_centers_])
print(np.where(kmeans.labels_ == 0)[0])
print(np.where(kmeans.labels_ == 1)[0])

kmeans = KMeans(n_clusters=2, max_iter=100).fit(chroma)
print([kmeans.labels_, kmeans.cluster_centers_])
print(np.where(kmeans.labels_ == 0)[0])
print(np.where(kmeans.labels_ == 1)[0])
