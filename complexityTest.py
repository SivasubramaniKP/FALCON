# Even simpler approach
import time
import falcon  # Original implementation 
import falconFFT
# Implementation with Ziggurat sampling

# Sign the same message with both implementations
message = b"Hello, Falcon!"

# Original FFT implementation
sk_fft = falcon.SecretKey(512)
start_time = time.time()
sig_fft = sk_fft.sign(message)
fft_time = time.time() - start_time

# Ziggurat implementation
sk_zig = falconFFT.SecretKey(512)
start_time = time.time()
sig_zig = sk_zig.sign(message)
zig_time = time.time() - start_time

print(f"FFT signing: {fft_time:.8f}s")
print(f"Ziggurat signing: {zig_time:.8f}s")
print(f"Speedup: {fft_time/zig_time:.8f}x")