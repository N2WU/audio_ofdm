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
from scipy import signal
from scipy.signal import max_len_seq as mls
import sounddevice as sd

global fs 
fs = 48000
sd.default.samplerate = fs
np.random.seed(6)

def tx_audio(audio_signal):
    input("Press any key to transmit")
    sd.play(audio_signal)
    sd.wait()
    return True

def tx_ofdm(move):
    bit_stream = [ord(c) for c in move]
    binary_list = ['{0:b}'.format(c) for c in bit_stream]
    binary_stream = np.zeros(7*len(binary_list))

    for i in range(len(binary_list)):
    # item = [int(x) for x in str(binary_list[i])]
        item = np.array(list(map(int, str(binary_list[i]))))
        while len(item) != 7:
            item = np.concatenate(([0],item))
        binary_stream[7*i:7*(i+1)] = item
    
    ## zero-pad
    numbits = 7056/np.size(binary_stream)
    bit_err_pad = np.zeros(7056) #gross
    for k in range(len(binary_stream)):
        bit_err_pad[int(k*numbits):int((k+1)*numbits)] = binary_stream[k]

    complex_data_stream = np.reshape(bit_err_pad, (-1, 2))*2 - 1
    complex_data_vec = (complex_data_stream[:,0] + 1j*complex_data_stream[:,1])/np.sqrt(2)

    # Parameters
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

    Nbits = 7
    preamble = np.random.randint(2, size=((2**Nbits) - 1)) * 2 - 1
    preamble = mls(8,taps=[8,7,2,1,0])[0] * 2 - 1
    #print("preamble is", preamble)

    pilot_index = np.arange(0,K,8)
    data_index = np.setdiff1d(np.arange(K),pilot_index)

    # Transmitter
    alphabet = [1+1j,1-1j,-1+1j,-1-1j]/np.sqrt(2)
    l_total = len(data_index)*NBlk
    l_diff = l_total - len(complex_data_vec)
    if l_diff > 0:
        complex_data_vec = np.append(complex_data_vec, (np.ones(l_diff) + 1j*np.ones(l_diff))/np.sqrt(2))
    image_data_vec = np.reshape(complex_data_vec,(-1,4))
    d = np.zeros([K,NBlk],dtype=np.complex64)

    u_info = []
    for i_blk in range(0,NBlk):
        d[pilot_index,i_blk] = np.random.choice(alphabet,len(pilot_index),replace=True)
        d[data_index,i_blk] = image_data_vec[:,i_blk]
        ifft_mod = K*Nsps*np.fft.ifft(d[:,i_blk],int(K*Nsps))
        ifft_mod = ifft_mod/np.max(np.abs(ifft_mod))
        if is_cp:
            cp = ifft_mod[-Ng:] #eliminated 1-index
            u_info = np.concatenate((u_info, cp, ifft_mod))
        else:
            zp = np.zeros(Ng)
            u_info = np.concatenate((u_info,ifft_mod,zp))
            #u_info = np.append(u_info, zp)

    ## Preamble Baseband ZERO-INDEX
    u_pre = signal.resample_poly(preamble*1.0, Nsps, 1)
    t_pre = np.arange(len(u_pre))/fs
    s_pre = np.real(u_pre * np.exp(1j*2*np.pi*fc*t_pre.T))
    s_pre = s_pre / np.max(np.abs(s_pre))

    ## Generate Baseband
    s_pause = np.zeros(Np)
    failsafe = np.zeros(int(1*fs))
    t_info = np.arange(len(u_info))/fs
    #print("u_info is, ", np.size(u_info))
    s_info = np.real(u_info * np.exp(1j*2*np.pi*f0*t_info.T))
    #print(" max s_info is, ", max(abs(s_info)))
    s_info = s_info / np.max(np.abs(s_info))
    s_frame = np.concatenate((failsafe,s_pre,s_pause,s_info,s_pause,s_pre,failsafe))
    #print("s_frame is, ", np.size(s_frame))
    s_frame /= np.max(np.abs(s_frame))
    s_frame = s_frame[:,None]
    tx_audio(s_frame)

    return True