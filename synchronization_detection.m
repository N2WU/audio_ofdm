clear all
rng(6);
bits = randi([0,1],1e3,1);

% 0phi, 1phi, 2phi, 3phi
symbol_stream = (reshape(bits,[],2) * 2) -1;



% real "stream" and imaginary stream

% 0,1 to (-1,1)
complex_stream = (symbol_stream(:,1) + 1i* symbol_stream(:,2))/sqrt(2);

% use variables, not magic numbers
% match symbol rate, imagine baud = 3kHz

b = 3e3;
fs = 48e3;
sps = fs/b;
fc = 10e3;
% expand signal -> upsample

upsampled_signal = repmat(complex_stream,[1 sps]);

upsampled_stream = reshape(upsampled_signal.',[],1);
% now it matches sampling rate, but still in baseband

t = (0:length(upsampled_stream) -1)/fs;
passband_signal = real(upsampled_stream.' .* exp(2*pi*fc*1i*t));
% only need real audio signal - imaginary is dsb

n = 128;
transform = fft(passband_signal,n);
test_signal = cos(2*pi*fc*t);
test_fft = fft(test_signal,n);
x_axis = linspace(-fs/2,fs/2,n);
plot(x_axis,abs(test_fft));
hold on
plot(x_axis, abs(transform));
hold off

% soundsc(passband_signal,fs);
% resample at correct fs
delay = 1e-1;
zeroarray = zeros(1,(fs * delay));
rx_signal = [zeroarray passband_signal];
t_rx = (0:length(rx_signal) -1)/fs;
baseband_signal = rx_signal .* exp(-2*pi*fc*1i*t_rx);
% peak detection: cross correlations
% if you want to be smart, cross-correlate with preamble (known) part of
% ofdm (known)
[r,delay] = xcorr(baseband_signal, upsampled_stream);

r = r(delay>=0);
delay = delay(delay>=0);

delay_axis = delay/fs;
plot(delay_axis,abs(r))

% in future, delay and sum (2 delays - watch number of samples)


