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
## resize image
image_stream = np.reshape(raw_image, -1) #transpose due to matlab quirk

# Generate bitstream
pre_size = 64
frame_size = 1024
space_size = 16
data_size = frame_size-pre_size - space_size
pre_bits = np.array([1,0,0,1,1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0])
pre_size = (len(pre_bits) + 1)/2

b = 3000
fs = 48000
sps = fs/b
fc = 10000
nsubcarriers = 128
npilot = 16
delta_f = b/nsubcarriers
delay = 0.1

complex_data_stream = np.reshape(image_stream, (-1, 2))*2 - 1
complex_data_vec_ur = (complex_data_stream[:,0] + 1j*complex_data_stream[:,1])/np.sqrt(2)

pre_bits = np.append(pre_bits,0)
pre_stream = np.reshape(pre_bits,[pre_size,2])*2 - 1
pre_symbols = (pre_stream[:,0] + 1j*pre_stream[:,1])/np.sqrt(2)
# Generate symbol stream

complex_data_vec = complex_data_vec_ur

symbol_mat = np.reshape(complex_data_vec, (nsubcarriers-npilot, -1))
pilot_symbols = np.random.randint(2, size=16)*2 - 1 ## bpsk

for k in np.size(symbol_mat,2):
    resource_grid = np.zeros(nsubcarriers)
    resource_grid(1:8:end) = pilot_symbols
    resource_grid(np.setdiff(1:nsubcarriers,1:8:nsubcarriers)) = symbol_mat[:,k]
    u_n = np.fft.ifft(resource_grid,nsubcarriers,1)
    cp = u_n[end-32:end]
    m[:,k] = np.append(cp,u_n)

u = np.reshape(m,-1)
pause = np.zeros(np.ceil(delay*fs/sps))
frame = np.append(pre_symbols, pause)
frame = np.append(frame, u)
frame = np.append(frame, pause)
frame = np.append(frame, pre_symbols)

# Transmit data
u_upsample = signal.resample(frame, int(sps*len(frame))) # so technically it repeats each bit 16 times
ts = np.arange(0,(len(u_upsample))/fs, 1/fs)

audio_signal = np.real(u_upsample * np.exp(1j*fc*2*np.pi*ts.T))
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

[r,delay] = np.correlate(baseband_signal,pre_symbols)
r = r[delay>=0]
delay = delay[delay>=0]

delay_axis = delay/fs

baseband_mat = np.reshape(baseband_signal,np.size(u_n))
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
