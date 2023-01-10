%% Image transmission via acoustic OFDM
% NP 2022/12/23
% Uses important code blocks from PSK counterpart
% Accounts for channel impairements, etc.

clear all
rng(6);

%% Load image

raw_image = double(imresize(imread("shostakovich_image.png"),[32 28]));
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
preamble_size = 64;
frame_size = 1024;
space_size = 16;
data_size = frame_size - preamble_size - space_size;
%preamble_bits = randi([0,1],1,preamble_data); % 16 blank symbols should be no transmission, not 0 data transmission

goldseq = comm.GoldSequence('FirstPolynomial','x^5+x^2+1', ...
    'SecondPolynomial','x^5+x^4+x^3+x^2+1', ...
    'FirstInitialConditions',[0 0 0 0 1], ...
    'SecondInitialConditions',[0 0 0 0 1], ...
    'Index',2,'SamplesPerFrame',2^5 - 1);
preamble_bits = goldseq();
preamble_size = (length(preamble_bits) + 1)/2;

% OFDM variables
b = 3e3; % bandwdith
fs = 48e3; % sampling freq
sps = fs/b; %samples per second (per symbol)
fc = 10e3; % carrier freq
%t = (0:length(complex_data_vec) -1)/fs
nsubcarriers = 128;
delta_f = b/nsubcarriers;

%pad 0 to sequence to get even number

complex_data_stream = reshape(image_stream, [length(image_stream)/2,2]).*2 - 1;
complex_data_vec_ur = (complex_data_stream(:,1) + 1i*complex_data_stream(:,2))./sqrt(2);
% gray code

% repeat with pilot bits

%pad 0 to sequence to get even number
preamble_bits = [preamble_bits; 0];

preamble_stream = reshape(preamble_bits, [preamble_size, 2]).*2 - 1;
preamble_symbols = (preamble_stream(:,1) + 1i*preamble_stream(:,2))./sqrt(2);

% build packet here - pilots, etc

%% Generate symbol stream
% Slot within resource grid?

% using imperfect data:
%round_int = mod(length(complex_data_vec_ur)+nsubcarriers,nsubcarriers);
complex_data_vec = [complex_data_vec_ur]; %.' zeros(nsubcarriers-round_int, 1).'].';
%t = (0:length(complex_data_vec) -1)/fs;

% 1. divide symbol stream d into k subcarriers
symbol_mat = reshape(complex_data_vec,nsubcarriers-16,[]);
pilot_symbols = randi([0 1], 16,1)*2 - 1;
% resource_grid = zeros(nsubcarriers,1); %evenly spaced symbol and pilot
% 2. feed parallel stream and 0s into idft block for n=0->Ns-1
% u_n = ifft(symbol_mat,nsubcarriers,1);
% 3. convert into serial again and filter with g(n)
for k=1:size(symbol_mat,2)
    resource_grid = zeros(nsubcarriers,1); %evenly spaced symbol and pilot
    resource_grid(1:8:end) = pilot_symbols;
    resource_grid(setdiff(1:nsubcarriers,1:8:nsubcarriers)) = symbol_mat(:,k);
    u_n = ifft(resource_grid,nsubcarriers,1);
    cp = u_n(end-32:end);
    m(:,k) = [cp; u_n];
end
u = reshape(m,[],1);
pause = zeros(ceil(1e-1 *fs/sps),1);
frame = [preamble_symbols; pause; u; pause ; preamble_symbols]; % preamble in time

%% Transmit data
% modulate to passband
u_upsample = resample(frame,sps,1);
ts = (0:length(u_upsample)-1)/fs;

audio_signal = real(u_upsample .* exp(1i*fc*2*pi*ts.'));
%sound(audio_signal,fs)

%% Channel Simulator
% delay
delay = 1e-1;
zeroarray = zeros(1,(fs * delay)).';
rx_signal = [zeroarray; audio_signal]; %4800 + 48000
t_rx = (0:length(rx_signal) -1)/fs;

% noise
%noise_signal = awgn(rx_signal,10);
%plot(t_rx(5e3:6e3),noise_signal(5e3:6e3));
noise_signal = rx_signal;
%% Receive Image
%1. Downshift and convert to parallel streams
rx_upsampled_signal = noise_signal .* exp(-2*pi*fc*1i*t_rx.'); %t_rx
baseband_signal = resample(rx_upsampled_signal, 1,sps);
% Delay impairement
[r,delay] = xcorr(baseband_signal, preamble_symbols);
r = r(delay>=0);
delay = delay(delay>=0);

delay_axis = delay/fs;
%plot(delay_axis,abs(r))

% Estimation
% Find the second-largest peak (hand-wave)
maxvec = maxk(abs(r),2);
max_location = find(abs(r) == maxvec(2));
baseband_signal_adj = baseband_signal(max_location:end); % equivalent to frame
% strip away unnecessary parts
rd_bit_len = length(preamble_symbols) + length(pause);
rx_packet = baseband_signal_adj(rd_bit_len+1:end-rd_bit_len); %equivalent to u

% step backwards from frame creation
% 1a. Convert to parallel stream
baseband_mat = reshape(rx_packet,size(m));

% 2. Remove Ng samples (CP or ZP) and feed into DFT
% requires for loop with resource grid
%{
for k=1:size(symbol_mat,2)
    resource_grid = zeros(nsubcarriers,1); %evenly spaced symbol and pilot
    resource_grid(1:8:end) = pilot_symbols;
    resource_grid(setdiff(1:nsubcarriers,1:8:nsubcarriers)) = symbol_mat(:,k);
    u_n = ifft(resource_grid,nsubcarriers,1);
    cp = u_n(end-32:end);
    m(:,k) = [cp; u_n];
end
%}
%for k=1:size(symbol_mat,2)
%    rs_grid_rx = zeros(nsubcarriers,1);
%    rs_grid_rx(1:8:end) = pilot_symbols;

eq = 0.416307/0.196317;
for k=1:size(symbol_mat,2)
    u_n_rx = baseband_mat(34:end,k) *eq; %strip away cp and "equalize"
    rs_grid_rx = fft(u_n_rx,nsubcarriers,1); %fft
    symbol_mat_rx(:,k) = rs_grid_rx(setdiff(1:nsubcarriers,1:8:nsubcarriers)); %slot only non-pilots
end

eq = 0.416307/0.196317;
figure(1)
plot(real(u_n), "+")
hold on
plot(real(u_n_rx), "o")
hold off
title("TX and RX Resource Grid for one Subcarrier")
legend("TX","RX")
xlabel("Symbol")
ylabel("Real Value")

post_fft = fft(baseband_mat, nsubcarriers);
post_fft = symbol_mat_rx;
%% Repack into symbol stream
% 4a. Parallel to serial conversion
rx_symbol_stream = reshape(post_fft,[],1).';

%% Repack into bitstream
% 4ab. Removing zeros
rx_symbol_stream = rx_symbol_stream(1:length(complex_data_vec_ur)).';
% some function to turn this into symbols? Just a coherence issue?

% 4b. Symbol Mapping
rx_symbol_mat = (sqrt(2)*[real(rx_symbol_stream) ; imag(rx_symbol_stream)] + 1)/2;
rx_image_stream = reshape(rx_symbol_mat.',[],1);%reshape(rx_symbol_mat.', [],1);

%% Format and display image

rx_image = reshape(rx_image_stream, size(raw_image));
figure(2)
imshow(rx_image.');