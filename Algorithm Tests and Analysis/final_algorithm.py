import scipy.io.wavfile
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from collections import deque, defaultdict

from os.path import dirname, join
AUDIO_LOC = join(dirname(dirname(__file__)), 'Audio Files')
FILES = [
         'Armin_van_Buuren_-_Ping_Pong_Original_Mix.wav',
         'This Is What It Feels Like feat. Trevor Guthrie (W&W Remix) (from zamzar to mp3).wav',
         'Bangarang - feat. Sirah.wav',
         'Breakn\' A Sweat.wav',
         'Summit - feat. Ellie Goulding.wav',
         'The Devil\'s Den.wav'
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
        self.modeled_actual_phase = None

    # Made iterable for plotting freq vs mag
    def __iter__(self):
        return self.freq_pair.__iter__()

    def next(self):
        return self.freq_pair.next()

    def __repr__(self):
        return 'Freq(' + str(self.freq) + ')'
        # return 'Freq(' + str(self.freq) + ', ' + str(self.phase) + ')'

    @property
    def freq_pair(self):
        return [self.freq, self.mag]

    @property
    def phase(self):
        return np.angle(self.amp)

    def model_actual_phase(self, DELAY):
        self.modeled_actual_phase = (self.phase-self.freq/60.0*2*pi*DELAY)%(2*pi)

def round_to(val, precision):
    return round(val / float(precision) * precision)

def get_beat_info(FILE, DELAY=10, RES_DESIRED=None):
    # User set parameters
    MAX_BPM = 160
    MIN_BPM = 80
    if RES_DESIRED == None:
        RES_DESIRED = float(1.0)
    LENGTH = 60.0/RES_DESIRED

    # Tuning parameters
    MAG_PERCENTILE_THRESHOLD = 98
    PEAK_SPLIT_TOLERAMCE = 5 # in BPM

    fs, data = scipy.io.wavfile.read(FILE)
    if fs*(DELAY+60.0/RES_DESIRED) > len(data):
        print('Audio not long enough. Cannot attain resolution of %s BPM.' % (RES_DESIRED))
        print('Highest resolution for song of length {:.4f} seconds with 0 DELAY is {:.4f} BPM'.format(float(len(data))/fs, fs*60.0/len(data)))
        return
    data = data[fs*DELAY:fs*(DELAY+LENGTH)]
    data = [channel0 for channel0, channel1 in data]
    plt.subplot(221)
    plt.plot(data[:fs*10])
    data = np.absolute(data, dtype='int64')
    data *= np.hanning(len(data))

    plt.subplot(222)
    plt.plot(data[:fs*10])

    xfft = np.fft.rfft(data)
    xfftmag = abs(xfft)
    xfftfreqs = np.fft.rfftfreq(len(data), d=1.0/fs)*60 # in bpm
    plt.subplot(223)
    plt.plot(xfftfreqs, xfft)

    filtered = [Frequency(xfftfreqs[i], xfft[i], xfftmag[i]) for i in range(MIN_BPM, int(MAX_BPM/RES_DESIRED)+1)]
    plt.subplot(224)
    plt.plot([freq_pair.freq for freq_pair in filtered], [freq_pair.mag for freq_pair in filtered])

    plt.show()

    mag_threshold = np.percentile([freq_pair.mag for freq_pair in filtered], MAG_PERCENTILE_THRESHOLD)
    peaks = []
    curr_peak = []
    skip_buffer = deque(maxlen=np.ceil(float(PEAK_SPLIT_TOLERAMCE)/RES_DESIRED))
    last_threshold_freq = MIN_BPM - PEAK_SPLIT_TOLERAMCE - 1
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
                if freq_pair.freq - last_threshold_freq > PEAK_SPLIT_TOLERAMCE:
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

    plt.subplot(111)
    plt.plot([freq_pair.freq for freq_pair in filtered], [freq_pair.mag for freq_pair in filtered])
    plt.show()

    fund_peak = peaks[0]
    likely_bpm_val = sum(freq_pair.freq*freq_pair.mag for freq_pair in fund_peak)/sum(freq_pair.mag for freq_pair in fund_peak)
    likely_bpm_val = round_to(likely_bpm_val, RES_DESIRED)
    likely_bpm = fund_peak[[freq_pair.freq for freq_pair in fund_peak].index(likely_bpm_val)]
    likely_bpm.model_actual_phase(DELAY)
    return likely_bpm
    # print('BPM of {0}+/-{1} with magnitude {2} and phase {3}.'.format(likely_bpm.freq, RES_DESIRED/2.0, likely_bpm.mag, likely_bpm.modeled_actual_phase))

for FILE in FILES[:]:
  for DELAY in [0]:
      print(FILE+': ' + str(get_beat_info(FILE, DELAY=DELAY)))