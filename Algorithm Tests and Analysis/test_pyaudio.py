import pyaudio
import sys
import scipy.io.wavfile

fs, audio_data = scipy.io.wavfile.read('This Is What It Feels Like feat. Trevor Guthrie (W&W Remix).wav', 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# define callback (2)
d = {'audio_data': audio_data}
def callback(in_data, frame_count, time_info, status):
    # data = wf.readframes(frame_count)
    data = d['audio_data'][:frame_count]
    d['audio_data'] = d['audio_data'][frame_count:]
    return (data, pyaudio.paContinue)

# open stream using callback (3)
stream = p.open(format=p.get_format_from_width(audio_data.itemsize),
                channels=len(audio_data[0]),
                rate=fs,
                output=True,
                stream_callback=callback)

print("-=-=-=-=-=-=-=-")

# start the stream (4)
stream.start_stream()

# wait for stream to finish (5)
while stream.is_active():
    pass

# stop stream (6)
stream.stop_stream()
stream.close()

# close PyAudio (7)
p.terminate()