%% Image transmission via acoustic OFDM
% NP 2022/12/23
% Uses important code blocks from PSK counterpart
% Accounts for channel impairements, etc.

clear all
rng(6);

%% Load image

raw_image = double(imread("stostakovich_image.png"));
image_stream = reshape(raw_image.', [], 1);
%image_stream_pad = [image_stream.' zeros((1.4e6 - length(image_stream)), 1).'].';

%% Generate bitstream (S-P Implementation)
% Format of data (subcarrier): [Pilot Data Data Pilot Data Data Pilot]
% Format of data (symbol index): [Pilot Data Pilot Pilot Pilot Data Pilot]
% -> Spatially and temporally-distributed pilot tones with guard interval
% -> Pilot tones are followed by 16 symbols of 0s
% Pilot Tones are 64 (48 + 16) symbols long, and data frames are 960 w/ 16
% blank bits (944 + 16)
% symbols (64 + 960 = 1024)
% Spatially distributed?

% add pilot bits here
pilot_size = 64;
frame_size = 1024;
space_size = 16;
pilot_data = pilot_size-space_size;
data_size = frame_size - pilot_size - space_size;
pilot_bits = randi([0,1],1,pilot_data); % 16 blank symbols should be no transmission, not 0 data transmission

% How are pilot bits added? Modulated via PSK?
% OFDM variables
b = 3e3; % bandwdith
fs = 48e3; % sampling freq
sps = fs/b; %samples per second
fc = 10e3; % carrier freq
%t = (0:length(complex_data_vec) -1)/fs
nsubcarriers = 128;
delta_f = b/nsubcarriers;

complex_data_stream = reshape(image_stream, [length(image_stream)/2,2]).*2 - 1;
complex_data_vec_ur = (complex_data_stream(:,1) + 1i*complex_data_stream(:,2))./sqrt(2);

% repeat with pilot bits

pilot_stream = reshape(pilot_bits, [pilot_data/2, 2]).*2 - 1;
pilot_symbols = (pilot_stream(:,1) + 1i*pilot_stream(:,2))./sqrt(2);

% build packet here - pilots, etc

%% Generate symbol stream
% Slot within resource grid?

% using imperfect data:
round_int = mod(length(complex_data_vec_ur)+nsubcarriers,nsubcarriers);
complex_data_vec = [complex_data_vec_ur.' zeros(nsubcarriers-round_int, 1).'].';
t = (0:length(complex_data_vec) -1)/fs;

% 1. divide symbol stream d into k subcarriers
symbol_mat = reshape(complex_data_vec,nsubcarriers,[]);
% 2. feed parallel stream and 0s into idft block for n=0->Ns-1
u_n = ifft(symbol_mat,nsubcarriers);
% 3. convert into serial again and filter with g(n)
u = reshape(u_n,[],1).';

%% Transmit data
% modulate to passband

audio_signal = real(u .* exp(1i*fc*2*pi*t));
%sound(audio_signal,fs)

%% Channel Simulator
% delay
delay = 1e-1;
zeroarray = zeros(1,(fs * delay));
rx_signal = [zeroarray audio_signal]; %4800 + 48000
t_rx = (0:length(rx_signal) -1)/fs;

% noise
%noise_signal = awgn(rx_signal,10);
%plot(t_rx(5e3:6e3),noise_signal(5e3:6e3));
noise_signal = audio_signal;
%% Receive Image

%1. Downshift and convert to parallel streams
baseband_signal = noise_signal .* exp(-2*pi*fc*1i*t); %t_rx

% Delay impairement
[r,delay] = xcorr(baseband_signal, pilot_symbols);
r = r(delay>=0);
delay = delay(delay>=0);

delay_axis = delay/fs;
% or, ignore all this and receive it without pilots or channel impairments?

% 1a. Convert to parallel stream
baseband_mat = reshape(baseband_signal,[],nsubcarriers);
% 2. Remove Ng samples (CP or ZP) and feed into DFT
post_fft = fft(baseband_mat.', nsubcarriers);

%% Repack into symbol stream
% 4a. Parallel to serial conversion
rx_symbol_stream = reshape(post_fft,[],1);

%% Repack into bitstream
% 4ab. Removing zeros
rx_symbol_stream = rx_symbol_stream(1:length(complex_data_vec_ur));
% some function to turn this into symbols? Just a coherence issue?

% 4b. Symbol Mapping
rx_symbol_mat = (sqrt(2)*[real(rx_symbol_stream) ; imag(rx_symbol_stream)] + 1)/2;
rx_image_stream = reshape(rx_symbol_mat.', [],1);

%% Format and display image

rx_image = reshape(rx_image_stream, size(raw_image));
imshow(rx_image);