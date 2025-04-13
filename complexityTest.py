# Even simpler approach
import time
import falcon_ACB  # Original implementation 
import falcon_FFT


# Before FFT
# Implementation with ACBgurat sampling

# Sign the same message with both implementations
message = b""" The project that you will build in this tutorial is available as Mini-Redis on GitHub. Mini-Redis is designed with the primary goal of learning Tokio, and is therefore very well commented, but this also means that Mini-Redis is missing some features you would want in a real Redis library. You can find production-ready Redis libraries on crates.io.

We will use Mini-Redis directly in the tutorial. This allows us to use parts of Mini-Redis in the tutorial before we implement them later in the tutorial.""" * 10000;

# Original FFT implementation
sk_fft = falcon_FFT.SecretKey(512)
start_time = time.time()
sig_fft = sk_fft.sign(message)
fft_time = time.time() - start_time

#ACB
sk_ACB = falcon_ACB.SecretKey(512)
start_time = time.time()
sig_ACB = sk_ACB.sign(message)
ACB_time = time.time() - start_time

# Get memory stats for FFT
# fft_current, fft_peak = tracemalloc.get_traced_memory()
# tracemalloc.stop()  # Stop the trace for FFT

# # Before ACB
# tracemalloc.start()  # Start new trace for ACB

# print(f"FFT signing: {fft_time:.8f}s")
# # Get memory stats for ACB
# acb_current, acb_peak = tracemalloc.get_traced_memory()
# tracemalloc.stop()

# # Calculate memory reduction percentage
# mem_reduction = (fft_peak - acb_peak) / fft_peak * 100 if fft_peak > 0 else 0

print(f"FFT signing: {fft_time:.8f}s")
print(f"ACB signing: {ACB_time:.8f}s")
print(f"Speedup: {fft_time/ACB_time:.8f}x")
# # Add memory results
# print(f"FFT peak memory: {fft_peak / 1024**2:.2f} MB")
# print(f"ACB peak memory: {acb_peak / 1024**2:.2f} MB")
# # print(f"Memory reduction: {mem_reduction:.2f}%")
