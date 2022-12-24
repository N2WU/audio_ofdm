% test code for symbol mapping (receive)
test_vec = [-1 + 1i, 1 + 1i, 1-1i, -1-1i]/sqrt(2);
split = ([real(test_vec) ; imag(test_vec)].' *sqrt(2) + 1)/2;
bit_vec = reshape(split.',[],1);