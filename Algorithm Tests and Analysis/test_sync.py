from __future__ import division
import scipy.io.wavfile
import numpy as np
from numpy import pi
from collections import deque
import itertools
import pyaudio
import matplotlib.pyplot as plt

FILES = ['This Is What It Feels Like feat. Trevor Guthrie (W&W Remix).wav',
         'This Is What It Feels Like feat. Trevor Guthrie (W&W Remix) (from mp3).wav',
         'This Is What It Feels Like feat. Trevor Guthrie (W&W Remix) (from zamzar to mp3).wav'
]

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

    def __init__(self, file_name, delay=6):
        self.file_name = file_name
        self.delay = delay
        self.fs, self.raw_data = scipy.io.wavfile.read(self.file_name)
        self.raw_data = self.remove_beginning_garbage(self.raw_data)
        self.__freq_data = self.get_freq_info()

    def __repr__(self):
        return "Song(" + self.file_name + ")"

    def get_clip(self):
        return self.raw_data[self.fs*self.delay:self.fs*(self.delay+self.CLIP_LENGTH)]

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
    def clip_phase(self):
        return self.__freq_data.clip_phase

    @property
    def start_sample(self):
        return self.full_phase/(2*pi)*60/self.freq*self.fs

    @property
    def ten_beats(self):
        return 10*60/self.freq*self.fs

    @property
    def one_beat(self):
        return 1*60/self.freq*self.fs

    def get_freq_info(self):
        if self.fs*(self.delay+60/self.RES_DESIRED) > len(self.raw_data):
            print('Audio not long enough. Cannot attain resolution of %s BPM.' % (self.RES_DESIRED))
            print('Highest resolution for song of length {:.4f} seconds with 0 delay is {:.4f} BPM'.format(len(self.raw_data)/self.fs, self.fs*60/len(self.raw_data)))
            return
        data = self.get_clip()
        data = [channel0 for channel0, channel1 in data]
        data = np.absolute(data, dtype='int64')
        data *= np.hanning(len(data))

        xfft = np.fft.rfft(data)
        xfftmag = abs(xfft)
        xfftfreqs = np.fft.rfftfreq(len(data), d=1/self.fs)*60 # in bpm

        filtered = [Frequency(xfftfreqs[i], xfft[i], xfftmag[i]) for i in range(self.MIN_BPM, self.MAX_BPM//self.RES_DESIRED+1)]
        mag_threshold = np.percentile([freq_pair.mag for freq_pair in filtered], self.MAG_PERCENTILE_THRESHOLD)
        peaks = []
        curr_peak = []
        skip_buffer = deque(maxlen=np.ceil(self.PEAK_SPLIT_TOLERANCE/self.RES_DESIRED))
        last_threshold_freq = self.MIN_BPM - self.PEAK_SPLIT_TOLERANCE - 1
        track = False
        for freq_pair in filtered:
            if freq_pair.mag > mag_threshold:
                track = True
                curr_peak.extend(skip_buffer)
                curr_peak.append(freq_pair)
                skip_buffer.clear()
                last_threshold_freq = freq_pair.freq
            else:
                if track == True:
                    if freq_pair.freq - last_threshold_freq > self.PEAK_SPLIT_TOLERANCE:
                        curr_peak.extend(skip_buffer)
                        skip_buffer.clear()
                        if curr_peak:
                            peaks.append(curr_peak)
                            curr_peak = []
                        track = False
                if len(skip_buffer) == skip_buffer.maxlen:
                    # EXTEND DEQUE CLASS TO AUTOMATICALLY SET TO ZERO?
                    skip_buffer[0].mag = 0
                skip_buffer.append(freq_pair)
        if curr_peak:
            curr_peak.extend(skip_buffer)
            peaks.append(curr_peak)
        else:
            for skipped in skip_buffer:
                skipped.mag = 0

        fund_peak = peaks[0]
        likely_bpm_val = sum(freq_pair.freq*freq_pair.mag for freq_pair in fund_peak)/sum(freq_pair.mag for freq_pair in fund_peak)
        likely_bpm_val = round_to(likely_bpm_val, self.RES_DESIRED)
        likely_bpm = fund_peak[[freq_pair.freq for freq_pair in fund_peak].index(likely_bpm_val)]
        likely_bpm.model_full_phase(self.delay)
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
# songs = [Song(file_name, delay) for file_name, delay in zip(FILES, delays)[::2]]

songs = [Song(file_name) for file_name in FILES[::2]]
print("Freq info computed")
for song in songs:
    print(song.freq_data)
    print(song.start_sample/song.fs)
    print("")

plt.plot(songs[0].raw_data[:3000])
plt.show()
plt.figure()
plt.plot(songs[1].raw_data[:3000])
plt.show()
# print songs[1].raw_data[:100]
# print songs[1].raw_data[songs[1].delay*songs[1].fs-1500:songs[1].delay*songs[1].fs-1400]


# working
# print songs[0].fs*songs[0].delay, songs[0].fs*songs[0].delay+songs[0].ten_beats
# print songs[1].ten_beats
# print "---"
# print 0, songs[0].start_sample+songs[0].ten_beats
# print songs[1].fs*songs[1].delay+songs[0].start_sample+

p = pyaudio.PyAudio()

# WORKS!!!!!
audio_buffer = songs[0].raw_data[songs[0].start_sample:songs[0].start_sample+1*songs[0].one_beat]
for beat_num in range(1, 34):
    song = songs[beat_num % 2]
    audio_buffer = np.concatenate((audio_buffer, song.raw_data[song.start_sample+beat_num*song.one_beat : song.start_sample+(beat_num+1)*song.one_beat]))

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