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
def scalrFGrid(N, dF):
    fr = np.arange(-N/2.0, N/2.0) * dF
    fx, fy = np.meshgrid(fr, -1*fr)
    f = np.sqrt( (fx**2) + (fy**2) )
    return f

def phaseScreen(N, f, fm, f0, cn2, dz, wvn):
    PDS = 0.033*cn2*np.exp( - (f/fm)**2  ) / np.power( (f**2 + f0**2), (11/3), )
    #PDS[N/2,N/2] = 0.0
    #Random array generation
    cn = np.random.randn(N,N) + 1j*np.random.randn(N,N)
    #Thing to inverse
    cn = cn * np.sqrt(PDS*dz*2*np.pi) * (2*np.pi*wvn / (N*df) )
    phz_tmp = np.fft.ifft2(np.fft.ifftshift(cn))
    return np.real(phz_tmp)