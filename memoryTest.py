# Add this to the top of your complexityTest.py file or create a new file with this content
import tracemalloc
import time
import sys

# The rest of your imports...

# Initialize memory tracking variables
fft_peak = 0
acb_peak = 0
mem_reduction = 0

# Run FFT signing with memory tracking
print("Running FFT signing with memory tracking...")
tracemalloc.start()
fft_start_time = time.time()

# Run your FFT algorithm here
# ...your existing FFT code...

fft_time = time.time() - fft_start_time
fft_current, fft_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Run ACB signing with memory tracking
print("Running ACB signing with memory tracking...")
tracemalloc.start()
acb_start_time = time.time()

# Run your ACB algorithm here
# ...your existing ACB code...

acb_time = time.time() - acb_start_time
acb_current, acb_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Calculate speed and memory improvements
speedup = fft_time / acb_time
mem_reduction = (fft_peak - acb_peak) / fft_peak * 100 if fft_peak > 0 else 0

# Print results
print(f"FFT signing: {fft_time:.6f}s")
print(f"ACB signing: {acb_time:.6f}s")
print(f"Speedup: {speedup:.6f}x")
print(f"FFT peak memory: {fft_peak / 1024**2:.2f} MB")
print(f"ACB peak memory: {acb_peak / 1024**2:.2f} MB")
print(f"Memory reduction: {mem_reduction:.2f}%")