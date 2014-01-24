import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import copy
from collections import deque

class Frequency():
	def __init__(self, freq, amp, mag=None):
		self.freq = freq
		self.amp = amp
		if mag == None:
			self.mag = abs(amp)
		else:
			self.mag = mag

	def __iter__(self):
		return self.freq_pair.__iter__()

	def next(self):
		return self.freq_pair.next()

	def __repr__(self):
		return 'Freq(' + str(self.freq_pair) + ')'

	@property
	def freq_pair(self):
		return [self.freq, self.mag]

def round_to(val, precision):
	return round(val / precision) * precision

# User set parameters
RES_DESIRED = 1 # in bpm
MAX_BPM = 300
MIN_BPM = 10
DELAY = 10 # in seconds

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
	# try/except and pickling just for testing algorithm
	try:
		with open('xfft', 'rb') as f:
			loaded_res, xfft, xfftmag, xfftfreqs, mag_filtered = pickle.load(f)
		if loaded_res != RES_DESIRED:
			xfft = None
	except:
		xfft = None

	if xfft == None:
		data = data[fs*DELAY:fs*(DELAY+60.0/RES_DESIRED)]
		data = [pair[0]/2.0+pair[1]/2.0 for pair in data]
		data = np.hanning(len(data))*data

		xfft = np.fft.rfft(data)
		xfftmag = abs(xfft)
		xfftfreqs = np.fft.rfftfreq(len(data), d=1.0/fs)*60 # in bpm
		
		mag_filtered = [Frequency(round_to(xfftfreqs[i], RES_DESIRED), xfft[i], xfftmag[i]) for i in range(len(xfftfreqs)) if MIN_BPM<=xfftfreqs[i]<=MAX_BPM]
		print('FFT computed')
		with open('xfft', 'wb') as f:
			pickle.dump([RES_DESIRED, xfft, xfftmag, xfftfreqs, mag_filtered], f)
	
	mag_threshold = np.percentile([freq_pair.mag for freq_pair in mag_filtered], MAG_PERCENTILE_THRESHOLD)

	peak_filtered = mag_filtered
	peaks = []
	curr_peak = []
	skip_buffer = deque(maxlen=np.ceil(float(PEAK_SPLIT_TOLERAMCE)/RES_DESIRED))
	last_threshold_freq = MIN_BPM - PEAK_SPLIT_TOLERAMCE - 1
	track = False
	for freq_pair in peak_filtered:
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

	bpms_to_plot = zip(*peak_filtered)
	plt.plot(*bpms_to_plot)
	plt.title('Percentile %s' % MAG_PERCENTILE_THRESHOLD)
	plt.show()

	fund_peak = peaks[0]
	likely_bpm_val = sum(freq_pair.freq*freq_pair.mag for freq_pair in fund_peak)/sum(freq_pair.mag for freq_pair in fund_peak)
	likely_bpm_val = round(likely_bpm_val / RES_DESIRED) * RES_DESIRED
	likely_bpm = fund_peak[[freq_pair.freq for freq_pair in fund_peak].index(likely_bpm_val)]
	## TODO: CORRECT PHASE FOR DELAY
	print('BPM of {0}+/-{1} with magnitude {2} and phase {3}.'.format(likely_bpm.freq, RES_DESIRED/2.0, likely_bpm.mag, np.angle(likely_bpm.amp)))

class Song():
	RES_DESIRED = 1 # bpm resolution
	MAX_BPM = 300 # max bpm to consider as the song's
	seconds_delay = 10 # in seconds
	window_length = 60.0 # in seconds

	def __init__(self, file_name):
		self.file_name = file_name
		self.fs, self.raw_data = scipy.io.wavfile.read(self.file_name)
		self.samples_delay = self.fs*self.seconds_delay
		self.windowed = self.raw_data[self.samples_delay:self.samples_delay+self.fs*self.window_length/self.RES_DESIRED]
		x = [pair[0] for pair in self.windowed]
		y = [pair[1] for pair in self.windowed]

		xfft = np.fft.rfft(x)
		xfftmag = abs(xfft)
		xfftfreqs = np.fft.rfftfreq(len(x), d=1.0/fs)*60 # in bpm
		paired = [[xfftfreqs[i], xfftmag[i]] for i in range(len(xfftmag))]
		print(xfftfreqs[:10])
		likely_bpm = filter(lambda pair: pair[0] < 300, sorted(paired, key=lambda pair: pair[1], reverse=True))[0][0]
		print(str(likely_bpm) + ' +/- %s' % (RES_DESIRED/2.0))
		likely_bpm_amp = xfft[np.where(xfftfreqs==likely_bpm)[0][0]]
		print(likely_bpm_amp)
		print(abs(likely_bpm_amp))
		print(np.angle(likely_bpm_amp))

		cutoff = MAX_BPM*window_length/60