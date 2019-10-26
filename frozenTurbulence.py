from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

#Functions
def k(La):
    '''Fucntion to calculate the spatial frequency'''
    return 2*np.pi / La

def dK(La, scale):
    return scale/La

def vonKarman(k, l0, L0, Cn2):
    print -(k**2)/((5.92/l0)**2)
    return 0.033*Cn2*np.exp( -(k**2)/(5.92/l0)**2 ) / np.power( k+(2*np.pi/L0), 11/6)

def pertubationRefractiveIndex(k, N, dk, dz, l0, L0, Cn):
    vK = vonKarman(k, l0, L0, Cn)
    return np.sqrt((2*np.pi/(N*dk))**2*2*np.pi*k**2*dz*vK)

def generateGrid(N):
    return np.random.normal(0,1,size = (N,N,2)).view(np.complex128)


def theta(k, N, dk, dz, l0, L0, Cn):
    return np.fft.ifft2(np.fft.fftshift(pertubationRefractiveIndex(k, N, dk, dz, l0, L0, Cn)*np.random.normal(0,1,size = (N,N,2)).view(np.complex128)))

res = 512 #Number of pixels N, in a N x N image
realSize = 3 #Diameter in m of the image
scale = realSize / res
wavelength = 0.005

Theta = theta(k(wavelength), res, dK(wavelength, scale), 1000, 0.01, 50, (10**(-14)))

PS = np.zeros((res,res))
for i in range(0, res):
    for j in range(0, res):
        PS[i,j] =np.real( Theta[i,j])
#rint np.angle(np.exp(1j*Theta[1,5]))




plt.figure()
im = plt.imshow(PS, cmap = cm.viridis)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.show()