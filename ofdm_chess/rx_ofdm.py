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

from matplotlib import image
from matplotlib import pyplot

from scipy import signal
from scipy.signal import max_len_seq as mls
import sounddevice as sd

global fs
fs = 48000
sd.default.samplerate = fs
sd.default.channels = 1

def decision(d):
     d = np.reshape(d,-1)
     b = [np.sign(np.real(d)), np.sign(np.imag(d))]
     return b

def rx_audio():
    # capture audio
    input("Press any key to receive")
    duration = 10  # seconds
    rx_signal = sd.rec(int(duration * fs))
    sd.wait()
    return rx_signal
    # return audio file to play with

def rx_ofdm():
    K = 1024
    B = 1000
    f0 = 5000
    fc = f0 + B/2
    fs = 48000

    is_cp = True

    Tg = 0.032
    Tp = Tg*2
    NBlk = 4
    Ng = int(np.ceil(Tg*fs))
    Np = int(np.ceil(Tp*fs))
    Nsps = fs/B
    Nb = Nsps*K + Ng
    Nf = Nb*NBlk

    pilot_index = np.arange(0,K,8)
    data_index = np.setdiff1d(np.arange(K),pilot_index)

    preamble = mls(8,taps=[8,7,2,1,0])[0] * 2 - 1
    u_pre = u_pre = signal.resample_poly(preamble*1.0, Nsps, 1)
    Nbits = 7
    r = rx_audio()
    t_r = np.arange(len(r))/fs

    v = r * np.exp(-1j*2*np.pi*fc*t_r.T)

    R = np.correlate(v,u_pre,"full")

    lags = np.arange(len(R))

    R = np.abs(R)
    R = R / np.max(R)

    peaks, _ = signal.find_peaks(R,0.7) ## sketchy

    # OFDM Processing
    v = r*np.exp(-1j*2*np.pi*f0*t_r.T)
    start = peaks[0] + Np + 1 # len(u_pre) + Np #1
    v_vec = np.arange(Nf) + start

    v_ofdm = v[v_vec.astype(int)]
    l = 64
    f_kk = np.fft.fft(np.eye(K))
    f_kl = np.concatenate((f_kk[:,:int(l/2)],f_kk[:,-int(l/2):]),axis=1)
    f_kpl = f_kl[pilot_index, :]
    d = np.zeros([K,NBlk],dtype=np.complex64)
    d_hat = np.zeros((K,NBlk),dtype=np.complex64)
    alphabet = [1+1j,1-1j,-1+1j,-1-1j]/np.sqrt(2)
    for i_blk in range(0,NBlk):
        d[pilot_index,i_blk] = np.random.choice(alphabet,len(pilot_index),replace=True)

    for i_blk in range(0,NBlk):
        v_ofdm_indices = np.arange((i_blk)*Nb,(i_blk+1)*Nb)
        v_blk_cp = v_ofdm[v_ofdm_indices.astype(int)]
        if is_cp:
            v_blk = v_blk_cp[Ng:] #this is correct
        else:
            v_blk = v_blk_cp
            v_blk[0:Ng] = v_blk[0:Ng] + v_blk[-Ng:]
        y_blk = np.fft.fft(v_blk,int(K*Nsps))
        y_blk = y_blk[0:K]

        H_est = y_blk[pilot_index] / d[pilot_index,i_blk]
        bl = 1/len(pilot_index) * f_kpl.conj().T@H_est
        H_indices = np.arange(len(H_est))
        H_vals = np.arange(K)
        H_interp = np.interp(H_vals,H_indices,H_est)
        H_interp = f_kl@bl

    d_hat[:,i_blk] = y_blk / H_interp

    d_hat_data = d_hat[data_index,:]
    d_data = d[data_index,:]
    bits_tx = np.array(decision(d_data))
    bits_rx = np.array(decision(d_hat_data))

    ber = np.sum(abs(bits_tx-bits_rx)) / bits_rx.size

    # Unpack RX Bits
    binary_stream = np.zeros(28)
    numbits = 7056/np.size(binary_stream)
    rx_bit_stream = np.zeros(np.size(binary_stream))

    bits_rx_vec = np.reshape(bits_rx, -1, order='F')
    l_t = len(bits_rx_vec)
    l_diff = 56
    ones_index = np.arange(l_t - l_diff*2,l_t) #something to check out
    text_index = np.setdiff1d(np.arange(l_t),ones_index)

    rx_text_bits = bits_rx_vec[text_index]
    rx_text_bits = rx_text_bits[0:7056]
    for k in range(np.size(binary_stream)):
        long_bits = rx_text_bits[int(k*numbits):int((k+1)*numbits)]
        rx_bit_stream[k] = round(np.mean(long_bits)+1)/2
    
    numchars = int(np.size(binary_stream)/7)
    char_rx = ''
    for k in range(numchars):
        bits = str(rx_bit_stream[k*7:(k+1)*7])
        char_rx += chr(int(bits[1:-1:3],2))

    return char_rx

