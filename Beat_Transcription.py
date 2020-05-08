
### First extract information relevent to rhythmic analysis

from IPython.display import Audio, display
import glob
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
from array import *
import soundfile as sf

thrown_samples = glob.glob('./Thrown/*.wav')

#t[0]: song filepath
#t[1]: percussive sample
#t[2]: harmonic sample
#t[3]: tempo
#t[4]: sample
#t[5]: onset times
#t[6]: beat times
#t[7]: onset frames
#t[8]: beat frames


t = list()

fs = 44100
for i in range(0,len(thrown_samples)):
    #create list to store song data
    sample_data = list()
    t.append(sample_data)
    t[i].append(thrown_samples[i])

    #load the song data
    sample, sr = librosa.load(thrown_samples[i], duration = 10)

    #set hop length
    hop_length = 512

    #separate song into harmonic and percussive components
    sample_harmonic, sample_percussive = librosa.effects.hpss(sample)

    #beat track on the percussive signal
    tempo, beat_times = librosa.beat.beat_track(y=sample_percussive,
                                                 sr=sr,
                                                 units = 'time')

    print(tempo)

    #beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=sample_percussive,
                                                 sr=sr)

    onset_frames = librosa.onset.onset_detect(sample_percussive,
                                              sr=sr,
                                              wait=1,
                                              pre_avg=1,
                                              post_avg=1,
                                              pre_max=1,
                                              post_max=1)

    onset_times = librosa.frames_to_time(onset_frames)


    t[i].append(sample_percussive)
    t[i].append(sample_harmonic)
    t[i].append(tempo)
    t[i].append(sample)
    t[i].append(onset_times)
    t[i].append(beat_times)
    t[i].append(onset_frames)
    t[i].append(beat_frames)

    clicks = librosa.clicks(frames=onset_frames, sr=sr, length=len(sample))


    if(t[i][0][9:-4] == 'thrown3'):
        sf.write('./Thrown/test_sounds/thrown_w_clicks' + str(i) + '.wav', clicks + sample, sr)


### Allow user to alter the tempo of the beat; outputs beat overlayed on harmonic portion of sample
### and the modified beat itself

import soundfile as sf
import math
import py

sr = 22050

def changeTempo(current_tempo, onset_times, desired_tempo):

    hi_hat, _ = librosa.load('./Thrown/test_sounds/sfx/closed_hi_hat.wav')
    drum_hit, _ = librosa.load('./Thrown/test_sounds/sfx/drum_hit.wav')

    desired_tempo_timing = 60/desired_tempo
    current_tempo_timing = 60/current_tempo

    scale_factor = desired_tempo_timing/current_tempo_timing

    onset_frames1 = librosa.time_to_frames(onset_times, sr=sr)

    clicks1 = librosa.clicks(frames = onset_frames1, sr=sr, click_duration = .01, length=len(t[2][4]), click = hi_hat)

    sf.write('./Thrown/test_sounds/thrown_w_beat.wav', clicks1 + t[1][2], sr)

    scaled_onset_times = []
    for i in range(0, len(onset_times)):
        scaled_onset_times.append(onset_times[i] * scale_factor)

    scaled_beat_frames = []
    for i in range(0, len(t[1][8])):
        scaled_beat_frames.append(t[1][8][i] * scale_factor)

    scaled_beat_times = []
    for i in range(0, len(t[1][6])):
        scaled_beat_times.append(t[1][6] * scale_factor)

    onset_frames2 = librosa.time_to_frames(scaled_onset_times, sr=sr)

    print(onset_frames2)

    scaled_sample = librosa.effects.time_stretch(t[1][2], 1/scale_factor)

    clicks2 = librosa.clicks(frames = onset_frames2, sr=sr, click_duration = .01, length=len(scaled_sample), click = hi_hat)
    clicks3 = librosa.clicks(frames = scaled_beat_frames, sr=sr, click_duration = .01, length=len(scaled_sample), click = drum_hit)

    librosa.output.write_wav('./Thrown/test_sounds/thrown_altered_beat.wav', clicks2 + clicks3 + scaled_sample, sr)
    sf.write('./Thrown/test_sounds/altered_beat.wav', clicks2 + clicks3, sr)

    plt.figure(figsize = (14,5))
    plt.title("Removing Percussive Sample, Scalability of Tempo Events")
    plt.vlines(scaled_onset_times, -1,1, color = 'c', linestyles = 'dashed')
    plt.vlines(scaled_beat_times, -1,1, color = 'y', linestyles = 'dashed')
    plt.ylim(-1,1)
    plt.xlim(0,5)


changeTempo(t[1][3], t[1][5], 60)


### displays harmonic and percussive stacked waveforms
def hpssDisplay(sample, tempo_events):

    #separate song into harmonic and percussive components
    sample_harmonic, sample_percussive = librosa.effects.hpss(sample)




    plt.figure(figsize = (14,5))
    plt.title("Decomposing the Signal into Percussive and Harmonic Elements")
    librosa.display.waveplot(sample_harmonic, sr, color = 'r', alpha = .6)
    librosa.display.waveplot(sample_percussive, sr, color = 'b', alpha = .6)
    plt.vlines(tempo_events, -1, 1, color='y', linestyles = 'dashed')
    plt.xlim(0,5)



hpssDisplay(t[1][4], t[1][6])



### display beat and onset events overlayed on the percussive sample



def beatEventsDisplay(percussive_sample, onset_times, beat_times):


    for i in range(0, len(beat_times)):

        for j in range(0, len(onset_times)):

            if isclose(onset_times[j], beat_times[i], abs_tol = .03):

                onset_times[j] = 0.0

    print(onset_times)


    plt.figure(figsize = (14,5))
    plt.title("Decomposing the Signal into Percussive and Harmonic Elements")
    librosa.display.waveplot(percussive_sample, sr, color = 'b', alpha = .6)
    plt.vlines(beat_times, -1, 1, color='y', linestyles = 'dashed')
    plt.vlines(onset_times, -1, 1, color = 'c', linestyles = 'dashed')
    plt.xlim(0,5)


beatEventsDisplay(t[1][1], t[1][5], t[1][6])


### Transcribe the beat based upon the temporal relationship between beat and onset events

from math import isclose


def transcribeBeat():

    print("Beat Transcription: ")
    print()

    for i in range(0,len(t)):

        if(t[i][0][9:-4] == 'thrown3'):
            print("sample: " + t[i][0][9:-4])

            thrown_tempo = t[i][3]
            thrown_onset_frames = t[i][7]
            thrown_beat_frames = t[i][8]

            beats_per_second = thrown_tempo/60
            secs_per_tempo_event = 1/beats_per_second

            thrown_onset_times = t[i][5]
            thrown_beat_times = t[i][6]

            #print(secs_per_tempo_event)
            #print(thrown_beat_times)
            #print(thrown_onset_times)

            tempo_event_count = 0
            total_events_count = 0

            k = 0
            for j in range(0,len(thrown_beat_times)):

                #print("j: " + str(j))
                #print("beat time: " + str(thrown_beat_times[j]))
                #print("current onset time: " + str(thrown_onset_times[k]))


                while(thrown_onset_times[k] < thrown_beat_times[j]):

                    #print("k: " + str(k))
                    #print("onset time: " + str(thrown_onset_times[k]))

                    if(isclose(thrown_onset_times[k], thrown_beat_times[j], abs_tol = .03)):
                        print("B:", end = "")
                        tempo_event_count+=1
                        total_events_count+=1

                    else:

                        if(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/4)*1, abs_tol = .03)):
                            print("3Q:", end = "")
                            total_events_count+=1
                        elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/4)*2, abs_tol = .03)):
                            print("2Q:", end = "")
                            total_events_count+=1
                        elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/4)*3, abs_tol = .03)):
                            print("1Q:", end = "")
                            total_events_count+=1


                        else:

                            if(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/8)*1, abs_tol = .03)):
                                print("7E:", end = "")
                                total_events_count+=1
                            elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/8)*2, abs_tol = .03)):
                                print("6E:", end = "")
                                total_events_count+=1
                            elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/8)*3, abs_tol = .03)):
                                print("5E", end = "")
                                total_events_count+=1
                            elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/8)*2, abs_tol = .03)):
                                print("4E", end = "")
                                total_events_count+=1
                            elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/8)*3, abs_tol = .03)):
                                print("3E", end = "")
                                total_events_count+=1
                            elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/8)*2, abs_tol = .03)):
                                print("2E", end = "")
                                total_events_count+=1
                            elif(isclose(thrown_onset_times[k], thrown_beat_times[j] - (secs_per_tempo_event/8)*3, abs_tol = .03)):
                                print("1E", end = "")
                                total_events_count+=1

                    k+=1

                    #print("updated onset time: " + str(thrown_onset_times[k]))

                if(isclose(thrown_onset_times[k], thrown_beat_times[j], abs_tol = .03)):
                        print("B:", end ="")
                        tempo_event_count+=1
                        total_events_count+=1
            print()
            print()

transcribeBeat()

    
