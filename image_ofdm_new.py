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

def decision(d):
     d = np.reshape(d,-1)
     b = [np.sign(np.real(d)), np.sign(np.imag(d))]
     return b

# Load image
## C:/Users/Nolan/Documents/GitHub/audio_ofdm/shostakovich_image_rs.png
## D:/pearc/Documents/GitHub/audio_ofdm/shostakovich_image_rs.png
raw_image = np.array(image.imread('D:/pearc/Documents/GitHub/audio_ofdm/shostakovich_image_rs.png'))
#raw_image = np.array(image.imread('C:/Users/Nolan/Documents/GitHub/audio_ofdm/shostakovich_image_rs.png'))
raw_image = raw_image[:,:,2]
## resize image
image_stream = np.reshape(raw_image, -1) #transpose due to matlab quirk
complex_data_stream = np.reshape(image_stream, (-1, 2))*2 - 1
complex_data_vec = (complex_data_stream[:,0] + 1j*complex_data_stream[:,1])/np.sqrt(2)

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
Ng = int(np.ceil(Tg*fs))
Np = int(np.ceil(Tp*fs))
Nsps = fs/B
Nb = Nsps*K + Ng
Nf = Nb*NBlk

Nbits = 7
preamble = np.random.randint(2, size=((2**Nbits) - 1)) * 2 - 1
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
    ifft_mod = ifft_mod/abs(max(ifft_mod))
    if is_cp:
        cp = ifft_mod[-Ng:] #eliminated 1-index
        u_info = np.concatenate((u_info, cp, ifft_mod))
    else:
        zp = np.zeros(Ng)
        u_info = np.concatenate((u_info,ifft_mod,zp))
        #u_info = np.append(u_info, zp)

## Preamble Baseband ZERO-INDEX
u_pre = signal.resample(preamble, int(Nsps*len(preamble)))
t_pre = np.arange(len(u_pre))/fs
s_pre = np.real(u_pre * np.exp(1j*2*np.pi*fc*t_pre.T))
s_pre = s_pre / max(abs(u_info))

## Generate Baseband
s_pause = np.zeros(Np)
failsafe = np.zeros(int(0.1*fs))
t_info = np.arange(len(u_info))/fs
#print("u_info is, ", np.size(u_info))
s_info = np.real(u_info * np.exp(1j*2*np.pi*f0*t_info.T))
#print(" max s_info is, ", max(abs(s_info)))
s_info = s_info / max(abs(s_info))
s_frame = np.concatenate((failsafe,s_pre,s_pause,s_info,s_pause,s_pre,failsafe))
#print("s_frame is, ", np.size(s_frame))

# sd.play(s_frame)

# Channel Simulator
hp = [1, 0.5, 0.2]
taup = 3/343 + np.array([0,0.003,0.007])
P = len(hp)

r = np.zeros(np.size(s_frame))
for p in range(0,P):
    r = r+hp[p]*np.roll(np.array(s_frame),int(np.ceil(taup[p]*fs)))

r = r + 0.01*np.random.randn(np.size(r))

# Estimation
## Delay
t_r = np.arange(len(r))/fs
r = s_frame
v = r * np.exp(-1j*2*np.pi*fc*t_r.T)

R = signal.correlate(v,u_pre)

lags = signal.correlation_lags(len(v),len(u_pre))
R = R[np.argwhere(lags<0)]
lags = np.array([x for x in lags if x < 0])

R = R.flatten()
R = np.absolute(R)
R = R / np.amax(R)

peaks, _ = signal.find_peaks(R,0.7) ## sketchy
# OFDM Processing
v = r*np.exp(-1j*2*np.pi*f0*t_r.T)
start = peaks[0] + len(u_pre) + Np # + 1
v_vec = np.arange(Nf) + start
#print(v_vec)
#print(len(v_vec))
#print(len(v))
v_ofdm = v[v_vec.astype(int)]

d_hat = np.zeros((K,NBlk),dtype=np.complex64)
for i_blk in range(0,NBlk):
    v_ofdm_indices = np.arange((i_blk)*Nb,(i_blk+1)*Nb)
    v_blk_cp = v_ofdm[v_ofdm_indices.astype(int)]
    if is_cp:
        v_blk = v_blk_cp[0:Ng] + v_blk[-Ng:] #this is incorrect
    else:
        v_blk = v_blk_cp
        v_blk[0:Ng] = v_blk[0:Ng] + v_blk[-Ng:]
    y_blk = np.fft.fft(v_blk,int(K*Nsps))
    y_blk = y_blk[0:K]

    H_est = y_blk[pilot_index] / d[pilot_index,i_blk]
    H_indices = np.arange(len(H_est))
    H_vals = np.arange(K)
    H_interp = np.interp(H_vals,H_indices,H_est)

    d_hat[:,i_blk] = y_blk / H_interp
print("v_blk length",len(v_blk[-Ng:]))
d_hat_data = d_hat[data_index,:]
d_data = d[data_index,:]
bits_tx = np.array(decision(d_data))
bits_rx = np.array(decision(d_hat_data))

ber = np.sum(abs(bits_tx-bits_rx)) / bits_rx.size

# Unpack RX Bits
bits_rx_vec = np.reshape(bits_rx, -1,order='F')
l_t = len(bits_rx_vec)
#ones_index_a = np.arange(l_t/2-l_diff, l_t/2)
#ones_index_b = np.arange(l_t-l_diff, l_t)
#ones_index = np.append(ones_index_a,ones_index_b) 
ones_index = np.arange(l_t - l_diff*2,l_t) #something to check out
image_index = np.setdiff1d(np.arange(l_t),ones_index)

rx_image_stream = bits_rx_vec[image_index]
rx_image = np.reshape(rx_image_stream, np.shape(raw_image))
display_image = pyplot.imshow(rx_image,cmap='gray',interpolation='nearest')
pyplot.show()
