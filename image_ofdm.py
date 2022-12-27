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

# Load image
raw_image = np.array(image.imread('C:/Users/Nolan/Documents/GitHub/audio_ofdm/shostakovich_image.png'))
image_stream = np.reshape(np.transpose(raw_image), -1) #transpose due to matlab quirk

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

complex_data_stream = np.reshape(image_stream, (-1, 2)) *2 - 1
complex_data_vec_ur = (complex_data_stream[:,1] + 1j*complex_data_stream[:,2])/np.sqrt(2)

pilot_stream = np.reshape(pilot_bits,[pilot_data/2,2])*2 - 1
pilot_symbols = (pilot_stream[:,1] + 1j*pilot_stream[:,2])/np.sqrt(2)
# Generate symbol stream
round_int = np.mod(len(complex_data_vec_ur)+nsubcarriers,nsubcarriers)
complex_data_vec = [complex_data_vec_ur, np.zeros(nsubcarriers-round_int,1)] #fix transposes
t = np.arrange(0,(len(complex_data_vec)-1)/fs, 1/fs)

symbol_mat = np.reshape(complex_data_vec, (nsubcarriers, -1))
u_n = np.ifft(np.transpose(symbol_mat), nsubcarriers)
u = np.reshape(u_n, -1) #transpose

# Transmit data
u_upsample = np.repeat(u, sps) # so technically it repeats each bit 16 times
ts = np.arrange(0,(len(u_upsample)-1)/fs, 1/fs)

audio_signal = np.real(u_upsample * np.exp(1j*fc*2*np.pi*ts))
# Channel simulator
## Delay
delay = 0.1
zeroarray = np.zeros(1,(fs*delay))
rx_signal = [zeroarray, audio_signal]
t_rx = np.arrange(0,(len(rx_signal)-1)/fs, 1/fs)
## Noise
noise_signal = np.awgn(rx_signal,10)
noise_signal = audio_signal

# Receive image
rx_upsampled_signal = noise_signal * np.exp(-2*np.py*fc*1j*ts)
# baseband_signal = np.resample(rx_upsampled_signal,1,sps)
baseband_signal = rx_upsampled_signal[::sps]
## Delay impairment
[r,delay] = np.xcorr(baseband_signal,pilot_symbols)
r = r[delay>=0]
delay = delay[delay>=0]

delay_axis = delay/fs
baseband_mat = np.reshape(baseband_signal,(-1,nsubcarriers))
post_fft = np.fft(np.tranpose(baseband_signal),nsubcarriers)

# Repack into symbol stream
rx_symbol_stream = np.reshape(post_fft,-1)

# Repack into bitstream
rx_symbol_stream = rx_symbol_stream[1:np.length(complex_data_vec_ur)]
## current hangup
rx_symbol_mat = [np.sqrt(2)*np.real(rx_symbol_stream) + 1, np.imag(rx_symbol_stream)] / 2
rx_image_stream = np.reshape(rx_symbol_mat, -1)

# Format and display image
rx_image = np.reshape(rx_image_stream, np.size(raw_image))
image.imshow(rx_image)
