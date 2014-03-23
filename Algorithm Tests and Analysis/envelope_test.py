import scipy.io.wavfile
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import cPickle as pickle
import copy
from collections import deque, defaultdict
import itertools

FILE = 'This Is What It Feels Like feat. Trevor Guthrie (W&W Remix).wav'

class Frequency():
	def __init__(self, freq, amp, mag=None):
		self.freq = freq
		self.amp = amp
		if mag == None:
			self.mag = abs(amp)
		else:
			self.mag = mag

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

def round_to(val, precision):
	return round(val / float(precision) * precision)

def run_test(FILE, selection, with_env, LENGTH=None, DELAY=10, Hanning=True):
	# User set parameters
	KNOWN_BPM = 130
	MAX_BPM = 300
	MIN_BPM = 10
	if LENGTH == None:
		RES_DESIRED = float(1.0) # in bpm
		LENGTH = 60.0/RES_DESIRED
	else:
		RES_DESIRED = 60.0/LENGTH

	# Tuning parameters
	MAG_PERCENTILE_THRESHOLD = 99
	PEAK_SPLIT_TOLERAMCE = 5 # in BPM

	fs, data = scipy.io.wavfile.read(FILE)
	if fs*(DELAY+60/RES_DESIRED) > len(data):
		print('Audio not long enough. Cannot attain resolution of %s BPM.' % (RES_DESIRED))
		print('Highest resolution for song of length {:.4f} seconds with 0 DELAY is {:.4f} BPM'.format(float(len(data))/fs, fs*60.0/len(data)))
	else:
		data = data[fs*DELAY:fs*(DELAY+LENGTH)]

		'''
		resulting phases of following 3 data selections are much closer with
		taking the absolute value than without
		'''
		if selection == 1:
			data = [pair[0] for pair in data]
		elif selection == 2:
			data = [pair[1] for pair in data]
		elif selection == 3:
			data = [x/2.0+y/2.0 for x,y in data]
		if with_env:
			data = np.absolute(data, dtype='int64')

		if Hanning == True:
			data *= np.hanning(len(data))

		xfft = np.fft.rfft(data)
		xfftmag = abs(xfft)
		xfftfreqs = np.fft.rfftfreq(len(data), d=1.0/fs)*60 # in bpm
		filtered = [Frequency(xfftfreqs[i], xfft[i], xfftmag[i]) for i in range(MIN_BPM, int(MAX_BPM/RES_DESIRED)+1)]
		# chosen_freq = filtered[int(KNOWN_BPM/RES_DESIRED)]
		# modeled_phase = (2.45057092903-pi+KNOWN_BPM/60.0*2*pi*DELAY)%(2*pi)-pi
		# if DELAY == None:
		# 	desired = [chosen_freq, filtered[int(KNOWN_BPM/RES_DESIRED)+1]]
		# else:
		# 	desired = [DELAY, chosen_freq, modeled_phase, modeled_phase-chosen_freq.phase, (modeled_phase-chosen_freq.phase)*60/(130*2*pi)]
		# return desired

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
					skip_buffer[0].mag = 0
				skip_buffer.append(freq_pair)
		if curr_peak:
			curr_peak.extend(skip_buffer)
			peaks.append(curr_peak)
		else:
			for skipped in skip_buffer:
				skipped.mag = 0

		# bpms_to_plot = zip(*filtered)
		# plt.plot(*bpms_to_plot)
		# plt.title('Percentile %s' % MAG_PERCENTILE_THRESHOLD)
		# plt.show()

		fund_peak = peaks[0]
		likely_bpm_val = sum(freq_pair.freq*freq_pair.mag for freq_pair in fund_peak)/sum(freq_pair.mag for freq_pair in fund_peak)
		likely_bpm_val = round_to(likely_bpm_val, RES_DESIRED)
		likely_bpm = fund_peak[[freq_pair.freq for freq_pair in fund_peak].index(likely_bpm_val)]
		foo = [(freq_pair.freq, np.angle(freq_pair.amp)) for freq_pair in fund_peak[2:-1]]
		print foo
		# delayed_phase = np.angle(likely_bpm.amp)
		# print(delayed_phase)
		# actual_phase = (delayed_phase+pi-likely_bpm.freq/60.0*(2*pi)*DELAY)%(2*pi)
		# print(actual_phase)
		# modeled_phase = (2.45057092903-pi+KNOWN_BPM/60.0*2*pi*DELAY)%(2*pi)-pi
		# TODO: CORRECT PHASE FOR DELAY
		print('BPM of {0}+/-{1} with magnitude {2} and phase {3}.'.format(likely_bpm.freq, RES_DESIRED/2.0, likely_bpm.mag, np.angle(likely_bpm.amp)))

run_test(FILE, 1, False, DELAY=10)
run_test(FILE, 2, False, DELAY=10)
run_test(FILE, 3, False, DELAY=10)
run_test(FILE, 1, True, DELAY=10)
run_test(FILE, 2, True, DELAY=10)
run_test(FILE, 3, True, DELAY=10)