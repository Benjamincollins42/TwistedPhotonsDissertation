from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


#L = 3 #Length
#N = 512 #Number of pixels
#l0 = 0.001
#L0 = 10
#wvl = 500e-9
#wvn = 2*np.pi / wvl
#dz = 100
#fm = 5.92/l0
#f0 = 2*np.pi / L0
#cn2 = 1e-14
#dx = L / N #Works when number is fudged but
#df = 1 / (N*dx)


#Scalar frequency grid
def freqRanges(N, L):
    '''Calculates the N xN frequency ranges of the axes.
    N = pixel width, L = length of grid side.'''
    dx = L/N
    xf = np.fft.fftfreq(N,dx)
    xf = np.fft.fftshift(xf) #Shift gets them in the right order for the fourier transforms
    yf = np.fft.fftfreq(N,dx)
    yf = np.fft.fftshift(yf)
    return xf, yf

def freq(N, L,):
    '''Calculates the N x N frequency space grid required for the power spectrum, also calculates the df.'''
    fRx, fRy = freqRanges(N, L)
    for i in range(0,N):
        for j in range(0,N):
            f[i,j] = np.sqrt(fRx[i]**2 + fRy[j]**2)
    df = fRx[2] - fRx[1]
    return f, df

def randCompGaussian(N):
    """Square random complex grid generator with a gaussian distribution of size N x N."""
    return np.random.randn(N,N) + 1j*np.random.randn(N,N)

def refractionPertubation(f, df, dz, cn2, L0, kl, wvn, N):
    """Computes the pertubation of the refractive index.
    dz = distance, cn2 = strength of turbulence, L0 = outer scale
    kl = 3.3/l0, wvn = wavenumber, N = pixel width"""
    PDS = 0.033*cn2*(f**2 + 1/(L0**2))**(-11/6) * np.exp(-(f/kl)**2)*(1 + 1.802*(f/kl) - 0.254*(f/kl)**(7/6))
    return np.sqrt(PDS*dz*2*np.pi) * (2*np.pi*wvn / (N*df) )
    
def phaseScreenGenInd(f, df, dz,cn2, L0, kl, wvn, N):
    """Phasescreen generation when you only need one."""
    PDS = 0.033*cn2*(f**2 + 1/(L0**2))**(-11/6) * np.exp(-(f/kl)**2)*(1 + 1.802*(f/kl) - 0.254*(f/kl)**(7/6))
    cn = randCompGaussian(N)
    PRI = np.sqrt(PDS*dz*2*np.pi) * (2*np.pi*wvn / (N*df) )
    return np.fft.ifftn(np.fft.fftshift(PRI * cn))

def phaseScreenGen(N, PRI):
    """Generates a phasescreen from its refractive index purtubation."""
    cn = randCompGaussian(N)
    return np.fft.ifftn(np.fft.fftshift(PRI * cn))

def phaseStackGenFroz(N, stackN, L, dz, cn2, L0, kl, wnv):
    """Generates stackN phasescreens of size N x N.
    N = pixel width, stackN = number of phasescreens, L = length of grid side,
    dz = distance, cn2 = strength of turbulence, L0 = outer scale, kl = 3.3/l0,
    wvn = wavenumber."""
    Freq = freq(N, L)
    PRI = refractionPertubation(Freq[0], Freq[1], dz, cn2, L0, kl, wvn, N)
    stack = np.zeros((N,N,stackN), dtype = 'complex128')
    stack[:,:,0] = phaseScreenGen(f, df, dz, cn2, L0, kl, wvn, N, PRI)
    for i in range(stackN):
        stack[:,:,i] = phaseScreenGen(f, df, dz, cn2, L0, kl, wvn, N, PRI)
    return stack

def alphaWind(modA, angleA, N, L):
    """Computes the alpha grid required for autoregressive phasescreen generation.
    modA = amount of phasescreen frequency retained each step,
    angleA = angle of 'wind', N = pixel width, L = length of grid side."""
    alpha = (np.zeros((N,N), dtype = 'complex128') + 1) * modA
    vx = np.cos(angleA)
    vy = np.sin(angleA)
    fRx, fRy = freqRanges(N, L)
    df = fRx[0]-fRx[1]
    for i in range(N):
        for j in range(N):
            alpha[i,j] = alpha[i,j] * np.exp(-1j*2*np.pi*df*(fRx[i]*vx + fRy[j]*vy))
    return alpha

def autoRegressionPhase(alpha, phase0, PRI, N):
    """Generates a new N x N phasescreen based on the previous one.
    alpha = regression thing, phase0 = original phascreen,
    PRI = refacrtive index pertubation, N = pixel width."""
    interior = alpha*np.fft.fftn(phase0) + np.fft.fftshift( np.sqrt(1 - abs(alpha**2))*PRI*randCompGaussian(N) )
    phase1 = np.fft.ifftn(interior)
    return phase1

def phaseStackGenAuto(N, stackN, L, modA, angleA, dz, cn2, L0, kl, wnv):
    """Generates stackN phasescreens of size N x N using autoreggresion.
    N = pixel width, stackN = number of phasescreens, L = length of grid side,
    modA = amount of phasescreen frequency retained each step,
    angleA = angle of 'wind', dz = distance, cn2 = strength of turbulence,
    L0 = outer scale, kl = 3.3/l0, wvn = wavenumber."""
    Freq = freq(N, L)
    PRI = refractionPertubation(Freq[0], Freq[1], dz, cn2, L0, kl, wvn, N)
    alpha = alphaWind(modA, angleA, N, L)
    stack = np.zeros((N,N,stackN), dtype = 'complex128')
    stack[:,:,0] = phaseScreenGen(Freq[0], Freq[1], dz, cn2, L0, kl, wvn, N, PRI)
    for i in range(stackN - 1):
        stack[:,:, i + 1] = autoRegressionPhase(alpha, stack[:,:,i], PRI, N)
    return stack

def propTurbulence(dz, U0, N, L, screen, wvn):
    """Propagates light through a phasescreen causing refraction using Fraunhofer diffraction.
    dz = distance propagated, U0 = input wave, N = pixel width, L = length of grid side,
    screen = phasescreen, wnn = wavenumber."""
    dx = L/N
    Um = np.fft.fftshift(np.fft.fftn(U0 * np.exp(1j*screen))*dx**2)
    xf, yf = freqRanges(N, L)
    for i in range(N):
        for j in range(N):
            Um[i,j] = np.exp(1j*(wvn**2 - (xf[j]*2*np.pi)**2 - (yf[i]*2*np.pi)**2)**0.5 * dz) * Um[i,j]
    Um = np.fft.ifftn(np.fft.fftshift(Um))/dx**2
    return Um

def propTurbulenceMulti(dz, u0, N, L, screenStack, wvn, zf, zi = 0.0):
    """Propagates light through multiple phasescreens causing refraction using Fraunhofer diffraction.
    dz = distance propagated, u0 = input wave, N = pixel width, L = length of grid side,
    screenStack = array of phasescreens, wnn = wavenumber, zf = final z coordinate,
    zi = inital z coordinate."""
    dx = L/N
    xf, yf = freqRanges(N, L)
    U = u0
    for k in range(int((zf-zi)/dz)):
        U = np.fft.fftshift(np.fft.fftn(U * np.exp(1j*screenStack[:,:,k]))*dx**2)
        for i in range(N):
            for j in range(N):
                U[i,j] = np.exp(1j*(wvn**2 - (xf[j]*2*np.pi)**2 - (yf[i]*2*np.pi)**2)**0.5 * dz) * U[i,j]
        U = np.fft.ifftn(np.fft.fftshift(U))/dx**2
    return U