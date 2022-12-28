#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nolan Pearce
# Created Date: 2022-12-26
# version ='1.0'
#----------------------------------------------------------------------------
"""tx_ofdm.py transmits the user-generated chess move signal. It is a function and not the main element."""
#----------------------------------------------------------------------------
import numpy as np
import sounddevice as sd
from scipy import signal

global fs 
fs = 48000
sd.default.samplerate = fs

def tx_audio(audio_signal):
    input("Press any key to transmit")
    sd.play(audio_signal)
    sd.wait()
    return True

def tx_ofdm(move):
    # Load stream
    ## Convert from ascii to bits
    byte_stream = np.fromstring(move, dtype='uint8', sep='')
    raw_bits = np.unpackbits(byte_stream)
    bit_stream = np.reshape(raw_bits, -1) #transpose due to matlab quirk

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

    complex_data_stream = np.reshape(bit_stream, (-1, 2))*2 - 1
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
    tx_audio(audio_signal)
    return True