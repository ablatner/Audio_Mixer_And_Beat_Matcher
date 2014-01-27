from math import pi
import matplotlib.pyplot as plt
import cPickle as pickle
import itertools
import numpy as np

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
		return [self.freq, self.phase_mod]

	@property
	def phase(self):
		return np.angle(self.amp)

def interpolate(data):
	for first, second in data:
		first.phase_mod = first.phase
		second.phase_mod = second.phase
		if first.phase > 0:
			first.phase_mod = -2*pi+first.phase_mod
		first.phase_mod = first.phase_mod + 2*pi
		second.phase_mod = second.phase_mod

	interpolated_points = [(130, first.phase_mod+(130-first.freq)*(second.phase_mod-first.phase_mod)/(second.freq-first.freq)) for first, second in data]
	for index, x in enumerate(data):
		x.insert(1, interpolated_points[index])

	to_plot = [zip(*line) for line in data]

	for line in to_plot:
		print line
		plt.plot(*line)
	plt.plot([130,130],[0,2*pi])
	plt.show()

with open('test_results', 'rb') as f:
	results = pickle.load(f)

print results
for result in itertools.chain.from_iterable(results.values()):
	print result

rectangular_windowed = results[False]
hanning_windowed = results[True]	

interpolate(rectangular_windowed)
interpolate(hanning_windowed)