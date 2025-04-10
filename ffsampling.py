"""This file contains important algorithms for Falcon.

- the Fast Fourier orthogonalization (in coefficient and FFT representation)
- the Fast Fourier nearest plane (in coefficient and FFT representation)
- the Fast Fourier sampling (only in FFT)
.
"""
from common import split, merge                         # Split, merge
from fft import add, sub, mul, div, adj                 # Operations in coef.
from fft import add_fft, sub_fft, mul_fft, div_fft, adj_fft  # Ops in FFT
from fft import split_fft, merge_fft, fft_ratio         # FFT
from samplerz import samplerz                           # Gaussian sampler in Z


def gram(B):
    """Compute the Gram matrix of B.

    Args:
        B: a matrix

    Format: coefficient
    """
    rows = range(len(B))
    ncols = len(B[0])
    deg = len(B[0][0])
    G = [[[0 for coef in range(deg)] for j in rows] for i in rows]
    for i in rows:
        for j in rows:
            for k in range(ncols):
                G[i][j] = add(G[i][j], mul(B[i][k], adj(B[j][k])))
    return G


def ldl(G):
    """
    Compute the LDL decomposition of G. Only works with 2 * 2 matrices.

    Args:
        G: a Gram matrix

    Format: coefficient

    Corresponds to algorithm 8 (LDL*) of Falcon's documentation,
    except it's in polynomial representation.
    """
    deg = len(G[0][0])
    dim = len(G)
    assert (dim == 2)
    assert (dim == len(G[0]))

    zero = [0] * deg
    one = [1] + [0] * (deg - 1)
    D00 = G[0][0][:]
    L10 = div(G[1][0], G[0][0])
    D11 = sub(G[1][1], mul(mul(L10, adj(L10)), G[0][0]))
    L = [[one, zero], [L10, one]]
    D = [[D00, zero], [zero, D11]]

    return [L, D]


def ldl_fft(G):
    """
    Compute the LDL decomposition of G. Only works with 2 * 2 matrices.

    Args:
        G: a Gram matrix

    Format: FFT

    Corresponds to algorithm 8 (LDL*) of Falcon's documentation.
    """
    deg = len(G[0][0])
    dim = len(G)
    assert (dim == 2)
    assert (dim == len(G[0]))

    zero = [0] * deg
    one = [1] * deg
    D00 = G[0][0][:]
    L10 = div_fft(G[1][0], G[0][0])
    D11 = sub_fft(G[1][1], mul_fft(mul_fft(L10, adj_fft(L10)), G[0][0]))
    L = [[one, zero], [L10, one]]
    D = [[D00, zero], [zero, D11]]

    return [L, D]


def ffldl(G):
    """Compute the ffLDL decomposition tree of G.

    Args:
        G: a Gram matrix

    Format: coefficient

    Corresponds to algorithm 9 (ffLDL) of Falcon's documentation,
    except it's in polynomial representation.
    """
    n = len(G[0][0])
    L, D = ldl(G)
    # Coefficients of L, D are elements of R[x]/(x^n - x^(n/2) + 1), in coefficient representation
    if (n > 2):
        # A bisection is done on elements of a 2*2 diagonal matrix.
        d00, d01 = split(D[0][0])
        d10, d11 = split(D[1][1])
        G0 = [[d00, d01], [adj(d01), d00]]
        G1 = [[d10, d11], [adj(d11), d10]]
        return [L[1][0], ffldl(G0), ffldl(G1)]
    elif (n == 2):
        # Bottom of the recursion.
        D[0][0][1] = 0
        D[1][1][1] = 0
        return [L[1][0], D[0][0], D[1][1]]


def ffldl_fft(G):
    """Compute the ffLDL decomposition tree of G.

    Args:
        G: a Gram matrix

    Format: FFT

    Corresponds to algorithm 9 (ffLDL) of Falcon's documentation.
    """
    n = len(G[0][0]) * fft_ratio
    L, D = ldl_fft(G)
    # Coefficients of L, D are elements of R[x]/(x^n - x^(n/2) + 1), in FFT representation
    if (n > 2):
        # A bisection is done on elements of a 2*2 diagonal matrix.
        d00, d01 = split_fft(D[0][0])
        d10, d11 = split_fft(D[1][1])
        G0 = [[d00, d01], [adj_fft(d01), d00]]
        G1 = [[d10, d11], [adj_fft(d11), d10]]
        return [L[1][0], ffldl_fft(G0), ffldl_fft(G1)]
    elif (n == 2):
        # End of the recursion (each element is real).
        return [L[1][0], D[0][0], D[1][1]]


def ffnp(t, T):
    """Compute the ffnp reduction of t, using T as auxilary information.

    Args:
        t: a vector
        T: a ldl decomposition tree

    Format: coefficient
    """
    n = len(t[0])
    z = [None, None]
    if (n > 1):
        l10, T0, T1 = T
        z[1] = merge(ffnp(split(t[1]), T1))
        t0b = add(t[0], mul(sub(t[1], z[1]), l10))
        z[0] = merge(ffnp(split(t0b), T0))
        return z
    elif (n == 1):
        z[0] = [round(t[0][0])]
        z[1] = [round(t[1][0])]
        return z


def ffnp_fft(t, T):
    """Compute the ffnp reduction of t, using T as auxilary information.

    Args:
        t: a vector
        T: a ldl decomposition tree

    Format: FFT
    """
    n = len(t[0]) * fft_ratio
    z = [0, 0]
    if (n > 1):
        l10, T0, T1 = T
        z[1] = merge_fft(ffnp_fft(split_fft(t[1]), T1))
        t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
        z[0] = merge_fft(ffnp_fft(split_fft(t0b), T0))
        return z
    elif (n == 1):
        z[0] = [round(t[0][0].real)]
        z[1] = [round(t[1][0].real)]
        return z


def ffsampling_fft(t, T, sigmin, randombytes):
    """Compute the ffsampling of t, using T as auxilary information.

    Args:
        t: a vector
        T: a ldl decomposition tree

    Format: FFT

    Corresponds to algorithm 11 (ffSampling) of Falcon's documentation.
    """
    n = len(t[0]) * fft_ratio
    z = [0, 0]
    if (n > 1):
        l10, T0, T1 = T
        z[1] = merge_fft(ffsampling_fft(split_fft(t[1]), T1, sigmin, randombytes))
        t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
        z[0] = merge_fft(ffsampling_fft(split_fft(t0b), T0, sigmin, randombytes))
        return z
    elif (n == 1):
        z[0] = [samplerz(t[0][0].real, T[0], sigmin, randombytes)]
        z[1] = [samplerz(t[1][0].real, T[0], sigmin, randombytes)]
        return z
# import numpy as np
# import struct
# import math

# # Ziggurat algorithm constants for normal distribution
# # These are pre-computed values for 128 rectangles
# ZIGGURAT_NOR_R = 3.6541528853610088
# ZIGGURAT_NOR_INV_R = 0.27366123732975828

# # The Ziggurat tables
# # These would normally be pre-computed, but for demonstration I'm providing a function
# def generate_ziggurat_tables(n=128):
#     """Generate the X and Y tables for the Ziggurat algorithm."""
#     m = np.zeros(n)
#     f = np.zeros(n)
    
#     # Compute X[i] and Y[i] values
#     m[0] = ZIGGURAT_NOR_R
#     f[0] = np.exp(-0.5 * m[0] * m[0])
    
#     for i in range(1, n):
#         # Find X[i] such that the area of rectangle is the same
#         m[i] = np.sqrt(-2.0 * np.log(np.exp(-0.5 * m[i-1] * m[i-1]) + 1.0/n))
#         f[i] = np.exp(-0.5 * m[i] * m[i])
    
#     return m, f

# # Generate tables
# X_ZIG, Y_ZIG = generate_ziggurat_tables()

# def zigguratRandom(randombytes):
#     """Generate a random 32-bit value using the provided random bytes function."""
#     data = randombytes(4)
#     return struct.unpack('<I', data)[0]

# def sample_ziggurat_normal(randombytes):
#     """
#     Sample from a standard normal distribution using the Ziggurat method.
#     """
#     while True:
#         # Generate a random unsigned integer and extract the index and sign
#         r = zigguratRandom(randombytes)
#         i = r & 0x7F  # 7 bits for table index (0-127)
#         sign = 1 if r & 0x80 else -1  # 1 bit for sign
        
#         # Extract a random value from (0,1)
#         u = (zigguratRandom(randombytes) & 0x7FFFFFFF) / 2147483648.0  # Using 31 bits
        
#         x = u * X_ZIG[i]
        
#         # Core rectangle acceptance (very common case, ~99% for i>0)
#         if i > 0 and abs(x) < X_ZIG[i-1]:
#             return sign * x
        
#         # Handle the base strip (i=0)
#         if i == 0:
#             # Sample from the tail of the distribution
#             xx = -np.log(u) / ZIGGURAT_NOR_R
#             if xx >= 0.5 * ZIGGURAT_NOR_R * ZIGGURAT_NOR_R:
#                 return sign * np.sqrt(2.0 * xx)
#         # Handle the wedges
#         else:
#             if u * (Y_ZIG[i-1] - Y_ZIG[i]) < np.exp(-0.5 * x * x) - Y_ZIG[i]:
#                 return sign * x

# def samplerz_ziggurat(center, sigma, sigmin, randombytes):
#     """
#     Sample from a discrete Gaussian distribution with the given center and standard deviation
#     using the Ziggurat method.
    
#     Replaces the existing samplerz function.
#     """
#     # Scale factor between target sigma and the base sigma (sigmin)
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
        
#         u = (zigguratRandom(randombytes) & 0x7FFFFFFF) / 2147483648.0
#         if u <= p:
#             return r

# # Now let's modify the ffsampling function to use our new sampler
# def ffsampling_ziggurat(t, T, sigmin, randombytes):
#     """Compute the ffsampling of t, using T as auxilary information.
    
#     This version uses Ziggurat sampling instead of FFT-based sampling.

#     Args:
#         t: a vector
#         T: a ldl decomposition tree
#         sigmin: minimum standard deviation
#         randombytes: random bytes generator function

#     Corresponds to a modified version of algorithm 11 (ffSampling) of Falcon's documentation.
#     """
#     n = len(t[0]) * fft_ratio  # Assuming fft_ratio is defined elsewhere
#     z = [0, 0]
#     if (n > 1):
#         l10, T0, T1 = T
#         z[1] = merge_fft(ffsampling_ziggurat(split_fft(t[1]), T1, sigmin, randombytes))
#         t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
#         z[0] = merge_fft(ffsampling_ziggurat(split_fft(t0b), T0, sigmin, randombytes))
#         return z
#     elif (n == 1):
#         # Here's where we use the new Ziggurat sampler instead of samplerz
#         z[0] = [samplerz_ziggurat(t[0][0].real, T[0], sigmin, randombytes)]
#         z[1] = [samplerz_ziggurat(t[1][0].real, T[0], sigmin, randombytes)]
#         return z