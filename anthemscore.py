from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from math import sin, cos, pi, ceil, log10


def nop():
	pass


def X(f: float, N: int, g: list, t: float):
	answer = 0
	answer += 1/N
	temp = 0
	for n in range(N):
		_ = g(t(n))*(cos(2*pi*f*t(n))-sin(2*pi*f*t(n))*1j)
		temp += _
	answer *= temp
	return answer

# open the wav file
samplerate, data = wavfile.read(input("Please type the file location: ").replace('\\', '/'))

#enable debug mode for graph output
debug = 0

# stereo to mono (only use 1st channel if more than 1)
if type(data[0]) != np.ndarray:
	data = np.array([[val] for val in data])
else:
	data = np.array([[val[0]] for val in data])

# define section length and overlap amount
section_length = 128
overlap = 0.5

section_length /= 1000

# get data like length, section count
seconds = len(data)/samplerate
samples_per_section = ceil(samplerate*section_length)

real_section_length = int(samples_per_section*(1+overlap))

sections = ceil(len(data)/real_section_length)

# generate the hann curve
hann_curve = [sin(n*pi/real_section_length) ** 2 for n in range(real_section_length)]

windows = {}

# move every section with overlap in an array
for section in range(sections):
	windows[section] = [list(x) for x in data[section*real_section_length:real_section_length*(1+section)]]

max_value_data = [0, 1]

# get max vol
for val in range(len(data)):
	if abs(data[val][0]) > max_value_data[1]:
		max_value_data = [val, abs(data[val][0])]

print(max_value_data) if debug else nop()

# multiply each section by hann curve and normalize to 1 max. this should be changed to be done for every section individually as currently only the loudest section will have a max amp of 1
for section in range(sections):
	for e in range(len(windows[section])):
		windows[section][e][0] *= hann_curve[e]/max_value_data[1]

# working till here

# generate rffts. not working, imaginary output is empty
ffts = {}

for section in range(sections):
	_sect = windows[section]

	fft_d_value = rfft(_sect)
	fft_freq = rfftfreq(samples_per_section, 1/samplerate)
	_ = []
	for i in range(len(fft_freq)):
		_.append(fft_freq[i])
	fft_freq = _
	print(fft_freq*(section == 0))

	if debug:
		x = list(range(len(fft_d_value)))
		y = [np.real(x) for x in fft_d_value]
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.scatter(x, y)
		plt.show()
		
		y = [np.imag(x) for x in fft_d_value]
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.scatter(x, y)
		plt.show()

	ffts[section] = [int(np.real(x) ** 2+np.imag(x) ** 2) for x in fft_d_value]

if debug:
	for i in range(sections):
		x = list(range(len(ffts[i])))
		y = ffts[i]
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.scatter(x, y)
		plt.show()


# generate a gradient with below colors
def gen_gradient():
	gradient = []

	list_of_colors = [(75, 0, 130), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 127, 0), (255, 0, 0)]

	no_steps = 100

	def LerpColour(c1,c2,t):
	    return (int(c1[0]+(c2[0]-c1[0])*t+0.5),int(c1[1]+(c2[1]-c1[1])*t+0.5),int(c1[2]+(c2[2]-c1[2])*t+0.5))

	for i in range(len(list_of_colors)-1):
	    for j in range(no_steps):
	        gradient.append(LerpColour(list_of_colors[i],list_of_colors[i+1],j/no_steps))

	return gradient

gradient = gen_gradient()

actual_size = len(gradient)

# generate image. working weirdly if at all
import PIL
image = PIL.Image.new(mode = "RGB", size=(len(ffts),actual_size))
for i in range(len(ffts)):
	for j in range(len(ffts[i])):
		try:
			image.putpixel((i,int(actual_size*j/len(ffts[i])+0.5)), gradient[int(ffts[i][j]*actual_size+0.5)])
		except:
			break

# show image
image.show()

# quit
input()