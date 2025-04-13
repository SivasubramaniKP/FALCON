import subprocess
import re
import statistics
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configuration
NUM_RUNS = 30  # Number of times to run the test
SCRIPT_PATH = "f:/INtern/FALCON/falcon.py/complexityTest.py"  # Path to your script
PYTHON_PATH = "C:/Users/SIVASU/AppData/Local/Programs/Python/Python312/python.exe"  # Path to your Python executable

# Initialize lists to store results
fft_times = []
acb_times = []
speedups = []

# Regular expressions to extract timing information
fft_pattern = r"FFT signing: (\d+\.\d+)s"
acb_pattern = r"ACB signing: (\d+\.\d+)s"
speedup_pattern = r"Speedup: (\d+\.\d+)x"

print(f"Starting automated performance testing with {NUM_RUNS} runs...")
print(f"Test script: {SCRIPT_PATH}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# Run the tests multiple times
for i in range(1, NUM_RUNS + 1):
    print(f"Run {i}/{NUM_RUNS}...", end=" ", flush=True)
    
    start_time = time.time()
    
    # Run the script and capture its output
    result = subprocess.run([PYTHON_PATH, SCRIPT_PATH], 
                           capture_output=True, 
                           text=True)
    
    run_time = time.time() - start_time
    
    # Extract timing information using regex
    output = result.stdout
    
    try:
        fft_time = float(re.search(fft_pattern, output).group(1))
        acb_time = float(re.search(acb_pattern, output).group(1))
        speedup = float(re.search(speedup_pattern, output).group(1))
        
        fft_times.append(fft_time)
        acb_times.append(acb_time)
        speedups.append(speedup)
        
        print(f"completed in {run_time:.2f}s (Speedup: {speedup:.4f}x)")
    except:
        print(f"Error parsing output: {output}")

# Calculate statistics
if speedups:
    avg_fft = statistics.mean(fft_times)
    avg_acb = statistics.mean(acb_times)
    avg_speedup = statistics.mean(speedups)
    median_speedup = statistics.median(speedups)
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    stddev_speedup = statistics.stdev(speedups) if len(speedups) > 1 else 0

    # Print results
    print("\n" + "=" * 50)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 50)
    print(f"Number of successful runs: {len(speedups)}/{NUM_RUNS}")
    print(f"Average FFT signing time: {avg_fft:.8f}s")
    print(f"Average ACB signing time: {avg_acb:.8f}s")
    print(f"Average speedup: {avg_speedup:.8f}x")
    print(f"Median speedup: {median_speedup:.8f}x")
    print(f"Range: {min_speedup:.4f}x - {max_speedup:.4f}x")
    print(f"Standard deviation: {stddev_speedup:.8f}")
    print("=" * 50)
    
    # Save results to file
    with open("performance_results.txt", "w") as f:
        f.write("FALCON PERFORMANCE TEST RESULTS\n")
        f.write(f"Test conducted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of runs: {len(speedups)}\n\n")
        f.write(f"Average FFT signing time: {avg_fft:.8f}s\n")
        f.write(f"Average ACB signing time: {avg_acb:.8f}s\n")
        f.write(f"Average speedup: {avg_speedup:.8f}x\n")
        f.write(f"Median speedup: {median_speedup:.8f}x\n")
        f.write(f"Minimum speedup: {min_speedup:.8f}x\n")
        f.write(f"Maximum speedup: {max_speedup:.8f}x\n")
        f.write(f"Standard deviation: {stddev_speedup:.8f}\n\n")
        
        f.write("Raw data:\n")
        f.write("Run,FFT_Time,ACB_Time,Speedup\n")
        for i, (fft, acb, spd) in enumerate(zip(fft_times, acb_times, speedups), 1):
            f.write(f"{i},{fft:.8f},{acb:.8f},{spd:.8f}\n")
    
    print(f"Detailed results saved to 'performance_results.txt'")
    
    # Create plots
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Speedup for each run
        plt.subplot(2, 2, 1)
        runs = list(range(1, len(speedups) + 1))
        plt.plot(runs, speedups, 'b-o', markersize=4)
        plt.axhline(y=avg_speedup, color='r', linestyle='--', label=f"Avg: {avg_speedup:.4f}x")
        plt.title('Speedup per Run', fontsize=14)
        plt.xlabel('Run Number', fontsize=12)
        plt.ylabel('Speedup (x)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Execution times comparison
        plt.subplot(2, 2, 2)
        plt.plot(runs, fft_times, 'r-o', label='FFT', markersize=4)
        plt.plot(runs, acb_times, 'g-o', label='ACB', markersize=4)
        plt.title('Execution Time per Run', fontsize=14)
        plt.xlabel('Run Number', fontsize=12)
        plt.ylabel('Time (s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Histogram of speedups
        plt.subplot(2, 2, 3)
        bins = np.linspace(min_speedup*0.95, max_speedup*1.05, 15)
        plt.hist(speedups, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=avg_speedup, color='r', linestyle='--', label=f"Avg: {avg_speedup:.4f}x")
        plt.axvline(x=median_speedup, color='g', linestyle='-.', label=f"Median: {median_speedup:.4f}x")
        plt.title('Speedup Distribution', fontsize=14)
        plt.xlabel('Speedup (x)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Box plot of execution times
        plt.subplot(2, 2, 4)
        box_data = [fft_times, acb_times]
        box = plt.boxplot(box_data, labels=['FFT', 'ACB'], patch_artist=True)
        
        # Fill boxes with colors
        colors = ['lightcoral', 'lightgreen']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            
        plt.title('Execution Time Distribution', fontsize=14)
        plt.ylabel('Time (s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add average time text
        plt.text(1, avg_fft*1.1, f"Avg: {avg_fft:.6f}s", 
                 horizontalalignment='center', color='darkred')
        plt.text(2, avg_acb*1.1, f"Avg: {avg_acb:.6f}s", 
                 horizontalalignment='center', color='darkgreen')
        
        plt.tight_layout()
        plt.savefig('performance_results.png', dpi=300)
        print(f"Performance visualization saved to 'performance_results.png'")
        
        # Create a summary chart
        plt.figure(figsize=(10, 6))
        labels = ['FFT Signing', 'ACB Signing']
        avg_times = [avg_fft, avg_acb]
        
        bars = plt.bar(labels, avg_times, color=['lightcoral', 'lightgreen'])
        plt.title(f'Average Execution Time Comparison\nSpeedup: {avg_speedup:.4f}x', fontsize=16)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        # Add time labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                    f'{height:.6f}s',
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('performance_summary.png', dpi=300)
        print(f"Summary visualization saved to 'performance_summary.png'")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("Make sure you have matplotlib installed: pip install matplotlib")
else:
    print("No valid results were collected.")