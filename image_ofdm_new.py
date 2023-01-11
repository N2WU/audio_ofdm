#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nolan Pearce
# Created Date: 2022-12-26
# version ='1.0'
#----------------------------------------------------------------------------
"""image_ofdm_new.py encodes, transmits, receives, and decodes a black and white image through acoustic ofdm using new oversampling techniques"""
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

# Parameters
K = 1024
B = 3000
f0 = 10000
fc = f0 + B/2
fs = 48000

is_cp = False

Tg = 0.032
Tp = Tg*2
NBlk = 4
Ng = np.ceil(Tg*fs)
Np = np.ceil(Tp*fs)
Nsps = fs/B
Nb = Nsps*K + Ng
Nf = Nb*NBlk

Nbits = 7
preamble = np.random.randint(2, size=2^Nbits - 1) * 2 - 1

K_vec = np.arange(1,K)
pilot_index = K_vec[::8]
data_index = np.setdiff1d(K_vec,pilot_index)

# Transmitter
alphabet = [1+1j,1-1j,-1+1j,-1-1j]/np.sqrt(2)
d = np.zeros([K,NBlk])

u_info = []
for i_blk in range(1,NBlk):
    d[:,i_blk] = np.random.choice(alphabet,K,replace=True)
    ifft_mod = K*Nsps*np.fft.ifft(d[:,i_blk],K*Nsps)
    ifft_mod = ifft_mod/abs(max(ifft_mod))
    if is_cp:
        cp = ifft_mod[len(ifft_mod)-Ng+1:-1]
        u_info = np.append(u_info,cp)
        u_info = np.append(u_info, ifft_mod)
    else:
        zp = np.zeros(Ng,1)
        u_info = np.append(u_info,ifft_mod)
        u_info = np.append(u_info, zp)

## Preamble Baseband
u_pre = signal.resample(preamble, int(Nsps*len(preamble)))
t_pre = np.arange(0,len(u_pre)-1)/fs
s_pre = np.real(u_pre * np.exp(1j*2*np.pi*fc*t_pre.T))
s_pre = s_pre / max(abs(u_info))

## Generate Baseband
s_pause = np.zeros(Np,1)
failsafe = np.zeros(0.1*fs,1)
t_info = np.arange(0,len(u_info)-1)/fs
s_info = np.real(u_info * np.exp(1j*2*np.pi*f0*t_info.T))
s_info = s_info / max(abs(s_info))
s_frame = np.append(s_pre,failsafe)
s_frame = np.append(s_frame,s_pause)
s_frame = np.append(s_frame,s_info)
s_frame = np.append(s_frame,s_pause)
s_frame = np.append(s_frame,s_pre)
s_frame = np.append(s_frame,failsafe)

# sd.play(s_frame)

# Channel Simulator
hp = [1, 0.5, 0.2]
taup = 3/343 + [0,0.03,0.07]
P = len(hp)

r = np.zeros(np.size(s_frame))
for p in range(1,P):
    r = r+hp[p]*np.roll(s_frame,np.ceil(taup[p]*fs))

r = r + 0.01*np.random.randn(np.size(r))

# Estimation
## Delay
t_u = np.arange(0,len(u)-1)/fs
v = r * np.exp(-1j*2*np.pi*fc*t_u.T)

[R,lags] = np.correlate(v,u_pre)
R[lags<0] = []
lags[lags<0] = []

R = abs(R)
R = R / max(R)

peaks = signal.find_peaks(R,0.7) ## sketchy

# OFDM Processing
v = r*np.exp(-1j*2*np.pi*f0*t_u.T)
start = peaks[0] + len(u_pre) + Np + 1
v_vec = np.arange(1,Nf-1)
v_ofdm = v[start + v_vec]

d_hat = np.zeros(K,NBlk)
for i_blk in range(1,NBlk):
    v_blk_cp = v_ofdm[(i_blk-1)*Nb+1:i_blk*Nb]
    if is_cp:
        v_blk = v_blk_cp[1:Ng] + v_blk[len(v_blk)-Ng+1:-1]
    else:
        v_blk = v_blk_cp
        v_blk[1:Ng] = v_blk[1:Ng] + v_blk[len(v_blk)-Ng+1:-1]
    y_blk = np.fft.fft(v_blk,K*Nsps)
    y_blk = y_blk[1:K]

    H_est = y_blk[pilot_index] / d[pilot_index,i_blk]
    H_interp = np.interp(H_est,K/len(pilot_index))

    d_hat[:,i_blk] = y_blk / H_interp

d_hat_data = d_hat[data_index,:]
d_data = d[data_index,:]

