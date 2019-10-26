from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

l0 = 0.01
L0 = 50
N= 512
wavelength = 0.5 * 10 ** (-6)
wavenumber = 2*np.pi / wavelength
dz = 1000
L = 3
km = 5.92/l0
k0 = 2*np.pi/L0
scale = L/N
Cn2 = 10 ** (-16)
dk = 5e-3
thetar = 0.0
cen = np.floor(N/2)

def complexGaussianArray():
    return np.random.normal(0,1,size = (N,N,2)).view(np.complex128)

def VKPDS(kappa):
    return 0.033 * Cn2 * np.exp( - (kappa/km)**2 ) / (kappa + k0) ** (11/6)

def PRI(kappa):
    return np.sqrt( 2*np.pi*dz*VKPDS(kappa) ) * 2*np.pi*wavenumber / (N*dk)

def kSpaceGrid3():
    fx = np.arange(-N/2,N/2) * dk
    fx, fy = np.meshgrid(fx, -1*fx)
    f = np.sqrt((fx**2) + (fy**2))
    return f

def Theta(gammaC):
    return np.fft.ifft2( np.fft.fftshift( gammaC ) )

kap = kSpaceGrid3()
cn = complexGaussianArray()
PS = np.zeros(shape = (N,N,2))
for i in range(0, N):
    for j in range(0, N):
        tempScale = PRI( kap[i,j] )
        PS[i,j,0] = tempScale * cn[i,j].real
        PS[i,j,1] = tempScale * cn[i,j].imag
PS = PS.view(np.complex128)
PS[np.int(cen),np.int(cen)]=0.0
resultA = np.real( Theta(PS.view(np.complex128)) )
#print PS[5,5]

Result = np.zeros((N,N))
for i in range(0, N):
    for j in range(0, N):
        Result[i,j] = resultA[i,j][0]

plt.figure()
im = plt.imshow( Result, cmap = cm.viridis)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.show()