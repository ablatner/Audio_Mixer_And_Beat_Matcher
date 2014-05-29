from __future__ import division, absolute_import, print_function
import scipy.io.wavfile, scipy.signal
import numpy as np
from numpy import pi
from collections import deque
import pyaudio

# Python 2 compatibility tweaks
try:
    range = xrange
except:
    pass

class SongMixer(object):
    def __init__():
        raise NotImplementedError

    def play():
        raise NotImplementedError

    def pause():
        raise NotImplementedError

    def stop():
        raise NotImplementedError

    def next_song():
        raise NotImplementedError

    def jump_to(position):
        raise NotImplementedError

    def prev_song():
        raise NotImplementedError

    def queue_playlist(playlist):
        raise NotImplementedError

    def queue_song(song):
        raise NotImplementedError

    def start_playlist():
        raise NotImplementedError

    def shuffle_playlist():
        raise NotImplementedError

class Frequency(object):
    def __init__(self, freq, amp, delay, mag=None):
        self._freq = freq
        self._delay = delay
        if mag != None:
            self._mag = mag
        else:
            self._mag = abs(amp)
        self._clip_phase = np.angle(amp)
        self._full_phase = (-self._clip_phase+self._freq/60*2*pi*delay)%(2*pi)

    def __repr__(self):
        return 'Freq(' + str(self._freq) + ')'

    @property
    def freq(self):
        return self._freq

    @property
    def mag(self):
        return self._mag

    @property
    def full_phase(self):
        return self._full_phase

class Song(object):

    MAX_BPM = 160
    MIN_BPM = 75
    RES_DESIRED = 1
    CHANNEL = 0

    # Tuning parameters
    MAG_PERCENTILE_THRESHOLD = 98
    PEAK_SPLIT_TOLERANCE = 5
    SILENCE_THRESHOLD = 200
    MIN_SILENCE_COUNT = 5

    # Calculated from set params
    CLIP_LENGTH = 60/RES_DESIRED

    def __init__(self, file_name, delay=6):
        self._name = file_name
        self._delay = delay
        self._fs, raw_data = scipy.io.wavfile.read(self._name)
        self._silence_length = self._track_intro_silence(raw_data)
        raw_data = raw_data[self._silence_length:]
        self._beg_freq_data = self._get_freq_info(raw_data)
        self._end_freq_data = self._get_freq_info(raw_data, end=True)

    def __repr__(self):
        return "Song(" + self._name + ")"

    # @property
    # def fs(self):
    #     return self._fs

    # @property
    # def freq_data(self):
    #     return self._freq_data

    # @property
    # def freq(self):
    #     return self._freq_data._freq

    # @property
    # def full_phase(self):
    #     return self._freq_data.full_phase

    # @property
    # def start_sample(self):
    #     return self.full_phase/(2*pi)*60/self.freq*self._fs

    # @property
    # def one_beat(self):
    #     return 1*60/self.freq*self._fs

    def _get_freq_info(self, raw_data, end=False):
        if end == False:
            raw_data = raw_data[self._fs*self._delay:
                                self._fs*(self._delay+60/self.RES_DESIRED)]
        else:
            raw_data = raw_data[-self._fs*(self._delay+60/self.RES_DESIRED):
                                -self._fs*self._delay]
        raw_data = [channels[self.CHANNEL] for channels in raw_data]
        raw_data = np.absolute(raw_data, dtype='int64')
        raw_data *= np.hanning(len(raw_data))

        xfft = np.fft.rfft(raw_data)
        xfftmag = abs(xfft)
        xfftfreqs = np.fft.rfftfreq(len(raw_data), d=1/self._fs)*60 # in bpm

        filtered = [Frequency(xfftfreqs[i], xfft[i], self._delay, xfftmag[i])
                for i in range(self.MIN_BPM, self.MAX_BPM//self.RES_DESIRED+1)]
        mag_threshold = np.percentile([freq_pair.mag for freq_pair in
                filtered], self.MAG_PERCENTILE_THRESHOLD)

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

        likely_bpm_val = sum(freq_pair.freq*freq_pair.mag for freq_pair in
                             peak)/sum(freq_pair.mag for freq_pair in peak)
        likely_bpm_val = round_to(likely_bpm_val, self.RES_DESIRED)
        likely_bpm = peak[[freq_pair.freq for freq_pair in peak].index(
                                                                likely_bpm_val)]
        return likely_bpm

    def _track_intro_silence(self, audio_data):
        counter = 0
        for sample in range(len(audio_data)):
            if abs(audio_data[sample][self.CHANNEL]) > self.SILENCE_THRESHOLD:
                counter += 1
                if counter > self.MIN_SILENCE_COUNT:
                    return max(sample-6, 0)
            else:
                counter = 0
        return 0

class AudioBuffer(object):
    def __init__(self, fs=48000):
        self.fs = fs
        self.data = deque()

    def add_audio(self, data):
        pass

    def queue_song(self, song):
        pass

    def read_buffer(in_data, frame_count, time_info, status):
        data = audio_buffer[current_frame.val:current_frame.val+frame_count]
        current_frame.val = current_frame.val+frame_count
        return (data, pyaudio.paContinue)

class SongQueue(object):
    def __init__(self):
        self.audio_buffer = AudioBuffer()
        # self.song_queue =

class AudioEvent(object):
    def __init__(self):
        pass

class CrossfadeEvent(AudioEvent):
    def __init__(self):
        pass

class CustomAudioEvent(AudioEvent):
    def __init__(self):
        pass

def round_to(val, precision=1):
    return round(val / precision) * precision

from os.path import dirname, join
AUDIO_LOC = join(dirname(dirname(__file__)), 'Audio Files')
FILES = [
         'Armin_van_Buuren_-_Ping_Pong_Original_Mix.wav',
         'This Is What It Feels Like feat. Trevor Guthrie (W&W Remix) (from zamzar to mp3).wav',
         'Bangarang - feat. Sirah.wav',
         'Breakn\' A Sweat.wav',
         'Summit - feat. Ellie Goulding.wav',
         'The Devil\'s Den.wav',
         '01 Sam\'s Town.wav',
         '03 When You Were Young.wav',
         '08 Bones.wav'
]
for index, file_name in enumerate(FILES):
    FILES[index] = join(AUDIO_LOC, file_name)

songs = [Song(file_name) for file_name in FILES[0:]]
print("Freq info computed")
for song in songs:
    print(song._name)
    print(song._beg_freq_data)
    print(song._end_freq_data)
    print(song._fs)

buffer = AudioBuffer()
buffer.add_audio(0)