from __future__ import division
import scipy.io.wavfile
import scipy.signal
import numpy as np
from numpy import pi
from collections import deque
import pyaudio
import matplotlib.pyplot as plt
from os.path import dirname, join

AUDIO_LOC = join(dirname(dirname(__file__)), 'Audio Files')

FILES = ['This Is What It Feels Like feat. Trevor Guthrie (W&W Remix).wav',
         'This Is What It Feels Like feat. Trevor Guthrie (W&W Remix) (from mp3).wav',
         'This Is What It Feels Like feat. Trevor Guthrie (W&W Remix) (from zamzar to mp3).wav',
         'Bangarang - feat. Sirah.wav'
]
for index, file_name in enumerate(FILES):
    FILES[index] = join(AUDIO_LOC, file_name)

class Frequency():
    def __init__(self, freq, amp, mag=None):
        self.freq = freq
        self.amp = amp
        if mag == None:
            self.mag = abs(amp)
        else:
            self.mag = mag
        self.full_clip_phase = None

    # Made iterable for plotting freq vs mag
    def __iter__(self):
        return self.freq_pair.__iter__()

    def next(self):
        return self.freq_pair.next()

    def __repr__(self):
        return 'Freq(' + str(self.freq) + ', ' + str(self.full_phase) + ', ' + str(self.clip_phase) + ')'
        # return 'Freq(' + str(self.freq) + ')'

    @property
    def freq_pair(self):
        return [self.freq, self.mag]

    @property
    def clip_phase(self):
        return np.angle(self.amp)

    def model_full_phase(self, delay):
        self.full_phase = (-self.clip_phase+self.freq/60*2*pi*delay)%(2*pi)

class Song():

    # later load from prefs?
    # User set parameters
    MAX_BPM = 300
    MIN_BPM = 10
    RES_DESIRED = 1 # has greatest affect on computation time, 1 is fine

    # Tuning parameters
    MAG_PERCENTILE_THRESHOLD = 98
    PEAK_SPLIT_TOLERANCE = 5
    SILENCE_THRESHOLD = 200
    MIN_SILENCE_COUNT = 5

    # Calculated from set params
    CLIP_LENGTH = 60/RES_DESIRED

    def __init__(self, file_name, delay=10):
        self.file_name = file_name
        self._delay = delay

        self.fs, self.raw_data = scipy.io.wavfile.read(self.file_name)
        self.raw_data = self.remove_beginning_garbage(self.raw_data)
        self.__freq_data = self.get_freq_info()

    def __repr__(self):
        return "Song(" + self.file_name + ")"

    @property
    def clip(self):
        return self.raw_data[self.fs*self._delay:self.fs*(self._delay+self.CLIP_LENGTH)]

    @property
    def freq_data(self):
        return self.__freq_data

    @property
    def freq(self):
        return self.__freq_data.freq

    @property
    def full_phase(self):
        return self.__freq_data.full_phase

    @property
    def start_sample(self):
        return self.full_phase/(2*pi)*60/self.freq*self.fs

    @property
    def one_beat(self):
        return 1*60/self.freq*self.fs

    def get_freq_info(self):
        if self.fs*(self._delay+60/self.RES_DESIRED) > len(self.raw_data):
            print('Audio not long enough. Cannot attain resolution of %s BPM.' % (self.RES_DESIRED))
            print('Highest resolution for song of length {:.4f} seconds with 0 delay is {:.4f} BPM'.format(len(self.raw_data)/self.fs, self.fs*60/len(self.raw_data)))
            return
        data = self.clip
        data = [channel0 for channel0, channel1 in data]
        data = np.absolute(data, dtype='int64')
        data *= np.hanning(len(data))

        xfft = np.fft.rfft(data)
        xfftmag = abs(xfft)
        xfftfreqs = np.fft.rfftfreq(len(data), d=1/self.fs)*60 # in bpm

        filtered = [Frequency(xfftfreqs[i], xfft[i], xfftmag[i]) for i in range(self.MIN_BPM, self.MAX_BPM//self.RES_DESIRED+1)]
        mag_threshold = np.percentile([freq_pair.mag for freq_pair in filtered], self.MAG_PERCENTILE_THRESHOLD)

        peak = []
        skip_buffer = deque(maxlen=np.ceil(self.PEAK_SPLIT_TOLERANCE/
                            self.RES_DESIRED))
        last_peak_freq = self.MIN_BPM - self.PEAK_SPLIT_TOLERANCE - 1
        track = False

        # skip_buffer must be empty when track is True
        for freq_pair in filtered:
            if freq_pair.mag > mag_threshold:
                if track == False:
                    track = True
                    peak.extend(skip_buffer)
                    skip_buffer.clear()
                peak.append(freq_pair)
                last_peak_freq = freq_pair.freq
            elif track == True:
                if freq_pair.freq - last_peak_freq > self.PEAK_SPLIT_TOLERANCE:
                    break
                peak.append(freq_pair)
            else:
                skip_buffer.append(freq_pair)

        likely_bpm_val = sum(freq_pair.freq*freq_pair.mag for freq_pair in peak)/sum(freq_pair.mag for freq_pair in peak)
        likely_bpm_val = round_to(likely_bpm_val, self.RES_DESIRED)
        likely_bpm = peak[[freq_pair.freq for freq_pair in peak].index(likely_bpm_val)]
        likely_bpm.model_full_phase(self._delay)
        return likely_bpm

    def remove_beginning_garbage(self, audio_data=None):
        if audio_data == None:
            audio_data = self.raw_data
        audio_data = audio_data[self.track_silence(audio_data):]
        return audio_data

    def track_silence(self, audio_data):
        counter = 0
        for sample in range(len(audio_data)):
            if abs(audio_data[sample][0]) > self.SILENCE_THRESHOLD or abs(audio_data[sample][1]) > self.SILENCE_THRESHOLD:
                counter += 1
                if counter > self.MIN_SILENCE_COUNT:
                    return max(sample-6, 0)
            else:
                counter = 0
        return 0

def round_to(val, precision=1):
    return round(val / precision) * precision

# delays = [0, 5.66632, .13387] # experimentally calculated to provide matching clip phases to within .0001 radians
# songs = [Song(file_name, delay) for file_name, delay in zip(FILES, delays)[:2]]
songs = [Song(file_name) for file_name in FILES[:3]]
print("Freq info computed")
for song in songs:
    print(song.file_name)
    print(song.freq_data)
    print(song.fs)
    print(song.start_sample/song.fs)
    print("")

p = pyaudio.PyAudio()

songs[1].raw_data = songs[1].raw_data[:songs[1].fs*50]
to_resample = songs[1].raw_data
plt.subplot(2,1,1)
length = 3
plt.plot(to_resample[:songs[1].fs*length])
print("Resampling")
songs[1].raw_data = scipy.signal.resample(to_resample, int(len(to_resample)*songs[0].fs/songs[1].fs))
songs[1].fs = songs[0].fs
print("Resampled")
songs[1].raw_data = songs[1].raw_data.astype(np.int16) ### CRITICAL LINE

# WORKS!!!!!
audio_buffer = songs[0].raw_data[songs[0].start_sample:songs[0].start_sample+22*songs[0].one_beat]
for beat_num in range(1, 4):
    song = songs[beat_num % 2]
    audio_buffer = np.concatenate((audio_buffer, song.raw_data[song.start_sample+22*beat_num*song.one_beat : song.start_sample+22*(beat_num+1)*song.one_beat]))
print(audio_buffer)
# MESSY AS FUCK
class MyInt():
    def __init__(self, num=0):
        self.val = num
current_frame = MyInt(0)

def callback(in_data, frame_count, time_info, status):
    data = audio_buffer[current_frame.val:current_frame.val+frame_count]
    current_frame.val = current_frame.val+frame_count
    return (data, pyaudio.paContinue)

stream = p.open(format=p.get_format_from_width(songs[0].raw_data.itemsize),
                channels=len(songs[0].raw_data[0]),
                rate=songs[0].fs,
                output=True,
                stream_callback=callback)

print("Writing to stream...")
stream.start_stream()
while stream.is_active():
    pass
stream.stop_stream()
stream.close()
p.terminate()