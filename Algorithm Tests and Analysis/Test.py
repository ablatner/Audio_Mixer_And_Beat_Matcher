import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import copy
from collections import deque, defaultdict
import itertools
from math import pi

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
		return 'Freq(' + str(self.freq) + ', ' + str(self.phase) + ')'

	@property
	def freq_pair(self):
		return [self.freq, self.mag]

	@property
	def phase(self):
		return np.angle(self.amp)

def round_to(val, precision):
	return round(val / precision) * precision

def run_test(LENGTH=None, DELAY=None, Hanning=True):
	# User set parameters
	KNOWN_BPM = 130
	RES_DESIRED = float(1.0) # in bpm
	MAX_BPM = 300
	MIN_BPM = 10
	if DELAY == None:
		DELAY = 10 # in seconds
	if LENGTH == None:
		LENGTH = 60.0/RES_DESIRED
	else:
		RES_DESIRED = float(60.0/LENGTH)

	# Tuning parameters
	MAG_PERCENTILE_THRESHOLD = 97
	PEAK_SPLIT_TOLERAMCE = 5 # in BPM

	# NUM_BANDS = 4
	# # From width*(sum(2^i, i, 0, n-2)+2^(n-2))=MAX_BPM
	# # Makes NUM_BANDS bands where each is twice the width of the preceding band
	# # The last band is centered on MAX_BPM
	# width = 4.0*MAX_BPM/(3*2**NUM_BANDS-4)
	# lower_band_edges = [np.floor(2**(i-1))*width for i in range(NUM_BANDS)]

	fs, data = scipy.io.wavfile.read('This Is What It Feels Like feat. Trevor Guthrie (W&W Remix).wav')
	if fs*(DELAY+60/RES_DESIRED) > len(data):
		print('Cannot attain resolution of %s BPM.' % (RES_DESIRED))
		print('Highest resolution for song of length {:.4f} seconds with 0 DELAY is {:.4f} BPM'.format(float(len(data))/fs, fs*60.0/len(data)))
	else:
		data = data[fs*DELAY:fs*(DELAY+LENGTH)]
		data = [pair[0] for pair in data]
		# data = [pair[0]/2.0+pair[1]/2.0 for pair in data]
		if Hanning == True:
			data = np.hanning(len(data))*data

		xfft = np.fft.rfft(data)
		xfftmag = abs(xfft)
		xfftfreqs = np.fft.rfftfreq(len(data), d=1.0/fs)*60 # in bpm
		
		# mag_filtered = [Frequency(round_to(xfftfreqs[i], RES_DESIRED), xfft[i], xfftmag[i]) for i in range(len(xfftfreqs)) if MIN_BPM<=xfftfreqs[i]<=MAX_BPM]
		mag_filtered = [Frequency(xfftfreqs[i], xfft[i], xfftmag[i]) for i in range(0, int(MAX_BPM/RES_DESIRED)+1)]
		chosen_freq = mag_filtered[int(KNOWN_BPM/RES_DESIRED)]
		modeled_phase = (2.45057092903-pi+KNOWN_BPM/60.0*2*pi*DELAY)%(2*pi)-pi
		if DELAY == None:
			desired = [chosen_freq, mag_filtered[int(KNOWN_BPM/RES_DESIRED)+1]]
		else:
			desired = [DELAY, chosen_freq, modeled_phase, modeled_phase-chosen_freq.phase, (modeled_phase-chosen_freq.phase)*60/(130*2*pi)]
		return desired
		
		# mag_threshold = np.percentile([freq_pair.mag for freq_pair in mag_filtered], MAG_PERCENTILE_THRESHOLD)

		# peak_filtered = mag_filtered
		# peaks = []
		# curr_peak = []
		# skip_buffer = deque(maxlen=np.ceil(float(PEAK_SPLIT_TOLERAMCE)/RES_DESIRED))
		# last_threshold_freq = MIN_BPM - PEAK_SPLIT_TOLERAMCE - 1
		# track = False
		# for freq_pair in peak_filtered:
		# 	if freq_pair.mag > mag_threshold:
		# 		track = True
		# 		curr_peak.extend(skip_buffer)
		# 		curr_peak.append(freq_pair)
		# 		skip_buffer.clear()
		# 		last_threshold_freq = freq_pair.freq
		# 	else:
		# 		if track == True:
		# 			if freq_pair.freq - last_threshold_freq > PEAK_SPLIT_TOLERAMCE:
		# 				curr_peak.extend(skip_buffer)
		# 				skip_buffer.clear()
		# 				if curr_peak:
		# 					peaks.append(curr_peak)
		# 					curr_peak = []					
		# 				track = False	
		# 		if len(skip_buffer) == skip_buffer.maxlen:
		# 			skip_buffer[0].mag = 0
		# 		skip_buffer.append(freq_pair)
		# if curr_peak:
		# 	curr_peak.extend(skip_buffer)
		# 	peaks.append(curr_peak)
		# else:
		# 	for skipped in skip_buffer:
		# 		skipped.mag = 0

		# bpms_to_plot = zip(*peak_filtered)
		# plt.plot(*bpms_to_plot)
		# plt.title('Percentile %s' % MAG_PERCENTILE_THRESHOLD)
		# plt.show()

		# fund_peak = peaks[0]
		# likely_bpm_val = sum(freq_pair.freq*freq_pair.mag for freq_pair in fund_peak)/sum(freq_pair.mag for freq_pair in fund_peak)
		# likely_bpm_val = round_to(likely_bpm_val, RES_DESIRED)
		# likely_bpm = fund_peak[[freq_pair.freq for freq_pair in fund_peak].index(likely_bpm_val)]
		# foo = [(freq_pair.freq, np.angle(freq_pair.amp)) for freq_pair in fund_peak[4:-1]]
		# print foo
		## TODO: CORRECT PHASE FOR DELAY
		# print('BPM of {0}+/-{1} with magnitude {2} and phase {3}.'.format(likely_bpm.freq, RES_DESIRED/2.0, likely_bpm.mag, np.angle(likely_bpm.amp)))

results = defaultdict(list)
for HANNING in [True]:
	# for LENGTH in np.linspace(59, 61, 11):
	for DELAY in np.linspace(0, 60, 31):
		results[HANNING].append(run_test(60, DELAY, Hanning=HANNING))
with open('test_results_delayed', 'wb') as f:
	pickle.dump(results, f)
print results
for result in itertools.chain.from_iterable(results.values()):
	print result