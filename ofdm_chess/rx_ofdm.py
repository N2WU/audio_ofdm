#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nolan Pearce
# Created Date: 2022-12-27
# version ='1.0'
#----------------------------------------------------------------------------
"""rx_ofdm.py receives the user-generated chess move signal. It is a function and not the main element."""
#----------------------------------------------------------------------------
import numpy as np

from scipy import signal

def rx_ofdm(rx_signal):
    b = 3000
    fs = 48000
    sps = fs/b
    fc = 10000
    nsubcarriers = 128
    data_length = 
    upsample_length = 
    ts = np.arange(0,upsample_length/fs, 1/fs)

    # Receive image
    rx_upsampled_signal = rx_signal * np.exp(-2*np.pi*fc*1j*ts)
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
    rx_symbol_stream = rx_symbol_stream[0:data_length]
    ## current hangup
    rx_symbol_real = (np.sqrt(2)*np.real(rx_symbol_stream) + 1)/2
    rx_symbol_imag = (np.sqrt(2)*np.imag(rx_symbol_stream) + 1)/2
    rx_symbol_mat = np.vstack((rx_symbol_real,rx_symbol_imag))
    rx_bit_stream = np.reshape(rx_symbol_mat, -1, order='F')
    rx_move = rx_bit_stream
    return rx_move
