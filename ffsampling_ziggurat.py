"""
Implementation of Ziggurat algorithm for Gaussian sampling in FALCON.
This module replaces the FFT-based sampling with more efficient Ziggurat sampling.
"""
import numpy as np
import struct
import math
from fft import fft_ratio, split_fft, merge_fft, add_fft, mul_fft, sub_fft, adj_fft

# Ziggurat algorithm constants
N_TABLES = 128  # Number of rectangles (power of 2 is more efficient)
ZIGGURAT_NOR_R = 3.6541528853610088  # Position of the rightmost rectangle
ZIGGURAT_NOR_INV_R = 0.27366123732975828  # 1/R

# Pre-computed tables for the Ziggurat algorithm
# X coordinates of the rectangles
X_ZIG = np.array([
    3.6541528853610088, 3.136445234382564, 2.8551791360620859, 2.6549676256770227,
    2.4984538639864519, 2.3701223815854456, 2.2604210337533648, 2.1639731018632544,
    2.0775412296046339, 1.9990588659627742, 1.9268937563019057, 1.8599624194040133,
    1.7974078782684295, 1.7384945698823212, 1.6826802945484658, 1.6295279595566089,
    1.5787079867863552, 1.5299970390359626, 1.4831543939611051, 1.4379871778939186,
    1.3943379966863296, 1.3520774248069411, 1.3110977719993853, 1.2713069661840363,
    1.2326170782257876, 1.1949544803737792, 1.1582489528960184, 1.1224324220504692,
    1.0874376401421585, 1.0532061789609187, 1.0196779452651579, 0.9867981185227845,
    0.9545169555044699, 0.9227897421732575, 0.8915764998773445, 0.8608398943244917,
    0.8305459939735207, 0.8006629357532311, 0.7711705020416036, 0.7420493631049833,
    0.7132823911939559, 0.6848534212271457, 0.6567476211075407, 0.6289507429437861,
    0.6014490700454118, 0.5742294730618064, 0.5472796443934513, 0.5205873963970918,
    0.4941404111121556, 0.4679266107756157, 0.4419339883668486, 0.4161506255183562,
    0.39056462878535271, 0.36516418527866066, 0.33993742885971555, 0.31487164215735803,
    0.28995419120143173, 0.26517142572619, 0.24050972564593997, 0.21595446120336708,
    0.19149064752736706, 0.16710326973781806, 0.14277537146062222, 0.11848900910345136,
    0.094225458862775762, 0.069953132142697367, 0.045636313944963991, 0.021234739149732071,
    0.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf
])

# Y coordinates (heights) of the rectangles
Y_ZIG = np.array([
    0.00026656726940199705, 0.0014867608343688444, 0.0041844775316104294, 0.0088633292822635134,
    0.015654180349193785, 0.024541092405865683, 0.035439302995579734, 0.048391375067883027,
    0.063380378107549957, 0.080406355592419825, 0.099488146634400836, 0.12065591974584019,
    0.14394499389539088, 0.16938738778716892, 0.19700999485263878, 0.22684007094821519,
    0.25889727167860304, 0.29320196841642712, 0.32976981865283065, 0.36862005066712039,
    0.40977382724773875, 0.45324955805421446, 0.49906347322256285, 0.54722917629926555,
    0.59775831940771214, 0.65066775001926511, 0.70598924302911168, 0.76375984806839696,
    0.82401864812894652, 0.88680646841814093, 0.95216051797535813, 1.0200276348471692,
    1.0903626137170237, 1.1631229551841663, 1.2382612252566027, 1.3157343867139348,
    1.3955041111444006, 1.4775362293656961, 1.5617997747562748, 1.6482672503771814,
    1.7369137292961183, 1.8277173583354135, 1.9206590992306075, 2.0157230501842104,
    2.1128964966889512, 2.2121705906133685, 2.3135392735244509, 2.4170010412499204,
    2.5225582533143764, 2.6302176921353213, 2.7399906518430504, 2.8518920883833278,
    2.9659420468221506, 3.0821642618461567, 3.2005862483404542, 3.3212390946357805,
    3.4441575879849823, 3.5693802638919147, 3.6969501566231621, 3.8269144219469961,
    3.9593244144870512, 4.0942352463175181, 4.2317069663050127, 4.3718057075929128,
    4.5146039702462167, 4.6601822759189319, 4.8086293510968811, 4.9600515874955713,
    5.1145728277642607, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
])

# def zig_random_32bit(randombytes):
#     """
#     Generate a random 32-bit integer using the provided random bytes function.
    
#     Args:
#         randombytes: Function that returns random bytes
    
#     Returns:
#         A random 32-bit unsigned integer
#     """
#     data = randombytes(4)
#     return struct.unpack('<I', data)[0]

# def zig_random_float(randombytes):
#     """
#     Generate a random float in [0,1) using the provided random bytes function.
    
#     Args:
#         randombytes: Function that returns random bytes
    
#     Returns:
#         A random float in [0,1)
#     """
#     # Use 31 bits for better precision
#     return (zig_random_32bit(randombytes) & 0x7FFFFFFF) / 2147483648.0

# def sample_ziggurat_normal(randombytes):
#     """
#     Sample from a standard normal distribution using the Ziggurat method.
    
#     Args:
#         randombytes: Function that returns random bytes
    
#     Returns:
#         A sample from N(0,1)
#     """
#     while True:
#         # Generate a random integer and extract the index and sign
#         r = zig_random_32bit(randombytes)
#         i = r & (N_TABLES - 1)  # Lower bits for table index
#         sign = 1 if r & 0x80 else -1  # Use a bit for sign
        
#         # Get a random value from [0,1)
#         u = zig_random_float(randombytes)
        
#         x = u * X_ZIG[i]
        
#         # Fast acceptance for the main case (i > 0 and x < X[i-1])
#         if i > 0 and abs(x) < X_ZIG[i-1]:
#             return sign * x
        
#         # Handle the base strip (i=0)
#         if i == 0:
#             # Rejection sampling from the tail
#             xx = -math.log(zig_random_float(randombytes)) / ZIGGURAT_NOR_R
#             if xx >= 0.5 * ZIGGURAT_NOR_R * ZIGGURAT_NOR_R:
#                 return sign * math.sqrt(2.0 * xx)
#         # Handle the wedges
#         else:
#             # This is where we test if we're under the Gaussian curve
#             y = Y_ZIG[i-1] + (Y_ZIG[i] - Y_ZIG[i-1]) * zig_random_float(randombytes)
#             if y < math.exp(-0.5 * x * x):
#                 return sign * x

# def samplerz(center, sigma, sigmin, randombytes):
#     """
#     Sample from a discrete Gaussian distribution with the given center and standard deviation.
    
#     Args:
#         center: Center of the distribution
#         sigma: Standard deviation
#         sigmin: Minimum standard deviation
#         randombytes: Function that returns random bytes
    
#     Returns:
#         A sample from the discrete Gaussian distribution
#     """
#     # If sigma is close to sigmin, we can use the Ziggurat sampler directly
#     # Otherwise, we need to scale the output
    
#     if sigma <= 1e-10:
#         return round(center)
    
#     sigma_ratio = sigma / sigmin
    
#     while True:
#         # Sample from standard normal using Ziggurat
#         x = sample_ziggurat_normal(randombytes)
        
#         # Scale to the target sigma
#         x = center + x * sigma_ratio
        
#         # Discrete rounding with probabilistic behavior
#         r = round(x)
        
#         # Accept with probability exp(-π·(x-r)²/σ²)
#         delta = x - r
#         p = math.exp(-math.pi * delta * delta / (sigma * sigma))
        
#         # Accept/reject
#         if zig_random_float(randombytes) <= p:
#             return r

# def ffsampling_ziggurat(t, T, sigmin, randombytes):
#     """
#     Compute the ffsampling of t, using T as auxiliary information.
#     This version uses Ziggurat sampling instead of FFT-based sampling.
    
#     Args:
#         t: a vector
#         T: a ldl decomposition tree
#         sigmin: minimum standard deviation
#         randombytes: random bytes generator function
    
#     Returns:
#         A vector z such that z approximates t
    
#     Corresponds to a modified version of algorithm 11 (ffSampling) of Falcon's documentation.
#     """
#     n = len(t[0]) * fft_ratio
#     z = [0, 0]
    
#     if n > 1:
#         l10, T0, T1 = T
        
#         # Recursive sampling for the second component
#         z[1] = merge_fft(ffsampling_ziggurat(split_fft(t[1]), T1, sigmin, randombytes))
        
#         # Compute t0' = t0 + (t1 - z1) · l10
#         t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
        
#         # Recursive sampling for the first component
#         z[0] = merge_fft(ffsampling_ziggurat(split_fft(t0b), T0, sigmin, randombytes))
        
#         return z
        
def zig_random_32bit(randombytes):
    """Generate a random 32-bit integer using the provided random bytes function."""
    return struct.unpack('<I', randombytes(4))[0]

def zig_random_float(randombytes):
    """Generate a random float in [0,1) using the provided random bytes function."""
    # Use 31 bits for better precision
    return (zig_random_32bit(randombytes) & 0x7FFFFFFF) / 2147483648.0

def sample_ziggurat_normal(randombytes):
    """Sample from a standard normal distribution using the Ziggurat method."""
    while True:
        # Generate a random integer and extract the index and sign in one step
        r = zig_random_32bit(randombytes)
        i = r & (N_TABLES - 1)  # Lower bits for table index
        sign = 1 if r & 0x80 else -1  # Use a bit for sign
        
        # Get a random value from [0,1)
        u = zig_random_float(randombytes)
        x = u * X_ZIG[i]
        
        # Fast acceptance for the main case (i > 0 and x < X[i-1])
        if i > 0 and x < X_ZIG[i-1]:
            return sign * x
        
        # Handle the base strip (i=0)
        if i == 0:
            # Rejection sampling from the tail
            xx = -math.log(zig_random_float(randombytes)) / ZIGGURAT_NOR_R
            if xx >= 0.5 * ZIGGURAT_NOR_R * ZIGGURAT_NOR_R:
                return sign * math.sqrt(2.0 * xx)
        # Handle the wedges
        elif y := Y_ZIG[i-1] + (Y_ZIG[i] - Y_ZIG[i-1]) * zig_random_float(randombytes):
            # Test if we're under the Gaussian curve
            if y < math.exp(-0.5 * x * x):
                return sign * x

def samplerz(center, sigma, sigmin, randombytes):
    """Sample from a discrete Gaussian distribution with the given center and standard deviation."""
    # Early return for negligible sigma
    if sigma <= 1e-10:
        return round(center)
    
    sigma_ratio = sigma / sigmin
    neg_pi_div_sigma_squared = -math.pi / (sigma * sigma)
    
    while True:
        # Sample from standard normal using Ziggurat
        x = center + sample_ziggurat_normal(randombytes) * sigma_ratio
        
        # Discrete rounding with probabilistic behavior
        r = round(x)
        
        # Accept with probability exp(-π·(x-r)²/σ²)
        delta = x - r
        p = math.exp(neg_pi_div_sigma_squared * delta * delta)
        
        # Accept/reject
        if zig_random_float(randombytes) <= p:
            return r

def ffsampling_ziggurat(t, T, sigmin, randombytes):
    """
    Compute the ffsampling of t, using T as auxiliary information.
    This version uses Ziggurat sampling instead of FFT-based sampling.
    """
    n = len(t[0]) * fft_ratio
    z = [0, 0]
    
    if n > 1:
        l10, T0, T1 = T
        
        # Recursive sampling for the second component
        z[1] = merge_fft(ffsampling_ziggurat(split_fft(t[1]), T1, sigmin, randombytes))
        
        # Compute t0' = t0 + (t1 - z1) · l10
        t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
        
        # Recursive sampling for the first component
        z[0] = merge_fft(ffsampling_ziggurat(split_fft(t0b), T0, sigmin, randombytes))
        
        return z
        
    elif n == 1:
        # Base case: use Ziggurat sampling
        if isinstance(T, list) and len(T) >= 3:
            sigma0, sigma1 = T[1], T[2]
        elif isinstance(T, tuple) and len(T) >= 2:
            sigma0, sigma1 = T[0], T[1]
        else:
            sigma0 = sigma1 = sigmin
            
        # Sample from discrete Gaussian
        z[0] = [samplerz(t[0][0].real, sigma0, sigmin, randombytes)]
        z[1] = [samplerz(t[1][0].real, sigma1, sigmin, randombytes)]
        
        return z 