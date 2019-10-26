from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

#Functions describing the beam
#Functions generating vaules that don't change over location
def zR(w0, La):
    '''Function to calculate the Rayleigh range'''
    return np.pi * w0 ** 2 / La

def wZ(w0, z, Zr):
    '''Function to calculate the spot size parameter'''
    return w0 * np.sqrt(1 + (z / Zr) ** 2)

def k(La):
    '''Fucntion to calculate the wavenumber'''
    return 2 * np.pi / La

def R(r, z, l, p, wz, k, zr):
    '''Fucntion to calculate the radial part of a beam'''
    return np.sqrt( (2*np.math.factorial(p)) / (np.pi*np.math.factorial(p + np.abs(l))) ) * (1/wz) * (r*np.sqrt(2)/wz) ** np.abs(l) * np.exp(-(r/wz)**2) * np.exp(-0.5j*k*z*r**2/(z**2 + zr**2))

def intensityGen(r, z, l, p, wz, k, zr, phi):
    '''Fucntion to calculate the intensity at a given point'''
    return 2 * np.abs(R(r, z, l, p, wz, k, zr)**2) * (1 + np.cos(2*l*phi))

#Constants
res = 512 #Number of pixels N, in a N x N image
realSize = 1 #Diameter in m of the image
scale = realSize / res
axialDistance = 100000
wavelength = 500 * 10 ** (-9)
waistRadius = 0.13
pQNumber = 0
lQNumber = 9

#Generating the arrays of distance and angle
distance = np.zeros((res,res))
angle = np.zeros((res,res))
for i in range(0, res):
    for j in range(0, res):
        x = (i - (res / 2) + 0.5)
        y = (j - (res / 2) - 0.5)
        distance[i,j] = np.sqrt(x**2 + y**2) * scale
        angle[i,j] = np.arctan(x / (y))

#Generating the intensity map
Intensity = np.zeros((res,res))
rayleighRange = zR(waistRadius, wavelength)
spotSizeParam = wZ(waistRadius, axialDistance, rayleighRange)
waveNumber = k(wavelength)

for i in range(0, res):
    for j in range(0, res):
        Intensity[i,j] = intensityGen(distance[i,j], axialDistance, lQNumber, pQNumber, spotSizeParam, waveNumber, rayleighRange, angle[i,j])

#Graph Plotting
plt.figure()
im = plt.imshow(Intensity, cmap = cm.viridis)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.show()