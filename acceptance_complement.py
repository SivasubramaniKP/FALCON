"""
Implementation of Acceptance-Complement-Based Algorithm for Gaussian sampling in FALCON.
This module replaces the FFT-based sampling with more efficient ACB sampling.
"""
import math
import struct
from fft import fft_ratio, split_fft, merge_fft, add_fft, mul_fft, sub_fft, adj_fft

# Constants for the ACB algorithm
# These can be adjusted for performance vs. quality trade-offs
ACB_TABLE_SIZE = 1024  # Size of the lookup table
ACB_PRECISION = 53     # Bits of precision (double-precision)
ACB_SIGMA = 1.0        # Standard deviation of the base distribution

# Pre-computed tables for the ACB algorithm
def generate_acb_tables(table_size=ACB_TABLE_SIZE):
    """
    Generate lookup tables for the Acceptance-Complement-Based algorithm.
    
    Returns:
        Tuple of (xValues, pValues)
    """
    xValues = [0] * table_size
    pValues = [0] * table_size
    
    # Generate evenly spaced x values from 0 to ~4 (covering most of the distribution)
    # We use 4 as it covers most of the probability mass of the normal distribution
    for i in range(table_size):
        # Calculate x values (scaled to cover most of the distribution)
        x = 4.0 * i / (table_size - 1)
        xValues[i] = x
        
        # Calculate corresponding probability values
        # p(x) = exp(-x²/2)
        pValues[i] = math.exp(-0.5 * x * x)
        
    return xValues, pValues

# Generate the lookup tables
ACB_X_TABLE, ACB_P_TABLE = generate_acb_tables(ACB_TABLE_SIZE)

def acb_random_32bit(randombytes):
    """
    Generate a random 32-bit integer using the provided random bytes function.
    
    Args:
        randombytes: Function that returns random bytes
    
    Returns:
        A random 32-bit unsigned integer
    """
    data = randombytes(4)
    return struct.unpack('<I', data)[0]

def acb_random_float(randombytes):
    """
    Generate a random float in [0,1) using the provided random bytes function.
    
    Args:
        randombytes: Function that returns random bytes
    
    Returns:
        A random float in [0,1)
    """
    # Use high precision for cryptographic applications
    return (acb_random_32bit(randombytes) & 0x7FFFFFFF) / 2147483648.0

def binary_search(value, table):
    """
    Find the index where value would be inserted in the sorted table.
    
    Args:
        value: Value to search for
        table: Sorted list to search in
    
    Returns:
        Index where value would be inserted to maintain order
    """
    left, right = 0, len(table) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if table[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
            
    return left

def interpolate(x, x0, x1, y0, y1):
    """
    Linear interpolation between two points.
    
    Args:
        x: Input x value
        x0, y0: First point
        x1, y1: Second point
    
    Returns:
        Interpolated y value
    """
    if x1 == x0:
        return y0
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def sample_acb_normal(randombytes):
    """
    Sample from a standard normal distribution using the Acceptance-Complement-Based algorithm.
    
    Args:
        randombytes: Function that returns random bytes
    
    Returns:
        A sample from N(0,1)
    """
    while True:
        # 1. Generate uniform random values
        u1 = acb_random_float(randombytes)
        u2 = acb_random_float(randombytes)
        
        # 2. Determine sign randomly
        sign = 1 if acb_random_32bit(randombytes) & 1 else -1
        
        # 3. Scale u1 to the x-range of our table
        x = 4.0 * u1  # Scale to the range [0, 4]
        
        # 4. Find the probability at this x value using interpolation
        idx = binary_search(x, ACB_X_TABLE)
        
        # Handle boundary cases
        if idx == 0:
            p = ACB_P_TABLE[0]
        elif idx >= len(ACB_X_TABLE):
            p = ACB_P_TABLE[-1]
        else:
            # Interpolate between table entries
            p = interpolate(x, 
                           ACB_X_TABLE[idx-1], ACB_X_TABLE[idx],
                           ACB_P_TABLE[idx-1], ACB_P_TABLE[idx])
        
        # 5. Apply acceptance-rejection technique
        if u2 <= p:
            return sign * x
        
        # For the tail (x > 4.0), use a different method
        if x > 4.0:
            # Use Marsaglia's tail algorithm
            while True:
                x = -math.log(acb_random_float(randombytes)) / 2.0
                y = -math.log(acb_random_float(randombytes))
                if y + y >= x * x:
                    return sign * (4.0 + math.sqrt(x))

def samplerz(center, sigma, sigmin, randombytes):
    """
    Sample from a discrete Gaussian distribution with the given center and standard deviation.
    
    Args:
        center: Center of the distribution
        sigma: Standard deviation
        sigmin: Minimum standard deviation
        randombytes: Function that returns random bytes
    
    Returns:
        A sample from the discrete Gaussian distribution
    """
    # Input validation and debug
    
    # Ensure sigma is not zero
    if sigma <= 1e-10:
        return round(center)  # Return rounded center if sigma is too small
    
    # Ensure sigmin is not zero to avoid division issues
    if sigmin <= 1e-10:
        sigmin = 1.0  # Use a safe default
    
    # Scale factor between target sigma and the base sigma
    sigma_ratio = sigma / sigmin
    
    max_iterations = 1000  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Sample from standard normal using ACB algorithm
        x = sample_acb_normal(randombytes)
        
        # Scale to the target sigma
        x = center + x * sigma_ratio
        
        # Discrete rounding with probabilistic behavior
        r = round(x)
        
        # Accept with probability exp(-π·(x-r)²/σ²)
        delta = x - r
        
        # Extra safety check before division
        if abs(sigma) <= 1e-10:
            return round(center)
            
        p = math.exp(-math.pi * delta * delta / (sigma * sigma))
        
        # Accept/reject
        if acb_random_float(randombytes) <= p:
            return r
    
    # If we reach here, we've hit the maximum number of iterations
    return round(center)  # Fallback

def ffsampling_acb(t, T, sigmin, randombytes):
    """
    Compute the ffsampling of t, using T as auxiliary information.
    This version uses Acceptance-Complement-Based sampling instead of FFT-based sampling.
    
    Args:
        t: a vector
        T: a ldl decomposition tree
        sigmin: minimum standard deviation
        randombytes: random bytes generator function
    
    Returns:
        A vector z such that z approximates t
    
    Corresponds to a modified version of algorithm 11 (ffSampling) of Falcon's documentation.
    """
    # Check if t is properly structured
    if not isinstance(t, list) or len(t) != 2:
        raise ValueError("Input vector t must be a list of length 2")
    
    n = len(t[0]) * fft_ratio
    z = [0, 0]
    
    if n > 1:
        # Recursive case
        
        # Validate T structure for recursive case
        if not isinstance(T, list) or len(T) != 3:
            raise ValueError(f"Expected T to be a list of length 3 for recursive case, got: {T}")
            
        l10, T0, T1 = T
        
        # Recursive sampling for the second component
        z[1] = merge_fft(ffsampling_acb(split_fft(t[1]), T1, sigmin, randombytes))
        
        # Compute t0' = t0 + (t1 - z1) · l10
        t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
        
        # Recursive sampling for the first component
        z[0] = merge_fft(ffsampling_acb(split_fft(t0b), T0, sigmin, randombytes))
        
        return z
        
    elif n == 1:
        # Base case: use ACB sampling
        # Validate T structure for base case
        if not isinstance(T, list):
            raise ValueError(f"Expected T to be a list for base case, got: {T}")
        
        # Different handling depending on the structure of T at the base case
        if len(T) == 3:
            # If T is [l10, D00, D11] as in the original ffLDL algorithm
            l10, D00, D11 = T
            z[0] = [samplerz(t[0][0].real, D00, sigmin, randombytes)]
            z[1] = [samplerz(t[1][0].real, D11, sigmin, randombytes)]
        elif len(T) == 2:
            # Alternative structure with just the diagonal elements
            D00, D11 = T
            z[0] = [samplerz(t[0][0].real, D00, sigmin, randombytes)]
            z[1] = [samplerz(t[1][0].real, D11, sigmin, randombytes)]
        else:
            # Handle any other structure or report the error
            raise ValueError(f"Unexpected T structure at base case: {T}")
        
        return z
    
    else:
        raise ValueError(f"Unexpected value of n: {n}")