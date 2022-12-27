#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nolan Pearce
# Created Date: 2022-12-26
# version ='1.0'
#----------------------------------------------------------------------------
"""image_ofdm.py encodes, transmits, receives, and decodes a black and white image through acoustic ofdm"""
#----------------------------------------------------------------------------

# Imports
import numpy as np

from matplotlib import image
from matplotlib import pyplot

from scipy import signal
import sounddevice as sd

# Load image
raw_image = np.array(image.imread('C:/Users/Nolan/Documents/GitHub/audio_ofdm/shostakovich_image.png'))
image_stream = np.reshape(raw_image, -1) #transpose due to matlab quirk

# Generate bitstream
pilot_size = 64
frame_size = 1024
space_size = 16
pilot_data = pilot_size-space_size
data_size = frame_size-pilot_size - space_size
pilot_bits = np.random.randint(2, size=pilot_data)

b = 3000
fs = 48000
sps = fs/b
fc = 10000
nsubcarriers = 128
delta_f = b/nsubcarriers

complex_data_stream = np.reshape(image_stream, (-1, 2))*2 - 1
complex_data_vec_ur = (complex_data_stream[:,0] + 1j*complex_data_stream[:,1])/np.sqrt(2)

pilot_stream = np.reshape(pilot_bits,[int(pilot_data/2),2])*2 - 1
pilot_symbols = (pilot_stream[:,0] + 1j*pilot_stream[:,1])/np.sqrt(2)
# Generate symbol stream
round_int = np.mod(len(complex_data_vec_ur)+nsubcarriers,nsubcarriers)
complex_data_vec = np.append(complex_data_vec_ur, np.zeros((1,int(nsubcarriers-round_int)), dtype=np.complex64)) #fix transposes
t = np.arange(0,(len(complex_data_vec))/fs, 1/fs)

symbol_mat = np.reshape(complex_data_vec, (nsubcarriers, -1))
u_n = np.fft.ifft(symbol_mat.T, nsubcarriers)
u = np.reshape(u_n, -1) #transpose

# Transmit data
u_upsample = signal.resample(u, int(sps*len(u))) # so technically it repeats each bit 16 times
ts = np.arange(0,(len(u_upsample))/fs, 1/fs)

audio_signal = np.real(u_upsample * np.exp(1j*fc*2*np.pi*ts))
#sd.play(audio_signal)
# Channel simulator
## Delay
delay = 0.1
zeroarray = np.zeros((1,int(fs*delay)))
rx_signal = [zeroarray, audio_signal]
t_rx = np.arange(0,(len(rx_signal))/fs, 1/fs)
## Noise
# noise_signal = np.awgn(rx_signal,10) #has to be fixed later
noise_signal = audio_signal

# Receive image
rx_upsampled_signal = noise_signal * np.exp(-2*np.pi*fc*1j*ts)
# baseband_signal = np.resample(rx_upsampled_signal,1,sps)
baseband_signal = signal.resample(rx_upsampled_signal, int(len(rx_upsampled_signal)/sps))
## Delay impairment
"""
[r,delay] = np.correlate(baseband_signal,pilot_symbols)
r = r[delay>=0]
delay = delay[delay>=0]

delay_axis = delay/fs
"""
baseband_mat = np.reshape(baseband_signal,(-1,nsubcarriers))
post_fft = np.fft.fft(baseband_mat,nsubcarriers).T #baseband_mat

# Repack into symbol stream
rx_symbol_stream = np.reshape(post_fft,-1)

# Repack into bitstream
rx_symbol_stream = rx_symbol_stream[0:len(complex_data_vec_ur)]
## current hangup
rx_symbol_real = (np.sqrt(2)*np.real(rx_symbol_stream) + 1)/2
rx_symbol_imag = (np.sqrt(2)*np.imag(rx_symbol_stream) + 1)/2
rx_symbol_mat = np.vstack((rx_symbol_real,rx_symbol_imag))
rx_image_stream = np.reshape(rx_symbol_mat, -1, order='F')

# Format and display image
rx_image = np.reshape(rx_image_stream, np.shape(raw_image))
display_image = pyplot.imshow(rx_image,cmap='gray',interpolation='nearest')
pyplot.show()
