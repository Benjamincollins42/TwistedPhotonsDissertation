from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

#Variables


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
Cn2 = 10 ** (-14)
dk = 5e2
thetar = 0.0
cen = np.floor(N/2)


#Functions


def complexGaussianArray():
    return np.random.normal(0,1,size = (N,N,2)).view(np.complex128)

def VKPDS(kappa):
    return 0.033 * Cn2 * np.exp( -(kappa/km)**2 ) / (kappa + k0) ** (11/6)

def PRI(kappa):
    return np.sqrt( 2*np.pi*dz*VKPDS(kappa) ) * 2*np.pi*wavenumber / (N*dk)

def Theta(gammaC):
    return np.fft.ifft2( np.fft.fftshift( gammaC ) ).view(np.complex128)

def phaseScreen(kappa):
    return np.real( Theta(kappa) )

def kSpaceGrid():
    distance = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            x = 1 /  (i - (N / 2) + 0.5) * scale 
            y = 1 /  (j - (N / 2) - 0.5) * scale 
            distance[i,j] = np.sqrt(x**2 + y**2)
    return distance

def kSpaceGrid2():
    distance = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            x = (i - (N / 2) + 0.5) * scale 
            y = (j - (N / 2) - 0.5) * scale 
            distance[i,j] = 1 / np.sqrt(x**2 + y**2)
    return distance

def kSpaceGrid3():
    fx = np.arange(-N/2,N/2) * dk
    fx, fy = np.meshgrid(fx, -1*fx)
    f = np.sqrt((fx**2) + (fy**2))
    return f


#Graphing

randComp = complexGaussianArray()
k = kSpaceGrid3()
PS = np.zeros((N,N,2))
print PRI( k[1,1] )
for i in range(0, N):
    for j in range(0, N):
        tempScale = PRI( k[i,j] )
        PS[i,j,0] = tempScale * randComp[i,j].real
        PS[i,j,1] = tempScale * randComp[i,j].imag
PS = PS.view(np.complex128)
PS[cen,cen] = 0.0 + 0.0j
Result = np.real( Theta(PS.view(np.complex128)) )

print Result[34,218]
Result2 = np.zeros((N,N))
for i in range(0, N):
    for j in range(0, N):
        Result2[i,j] = Result[i,j][0]


plt.figure()
im = plt.imshow( Result2, cmap = cm.viridis)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.show()