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

def intensityGenR(r, z, l, p, wz, k, zr, phi):
    '''Fucntion to calculate the intensity at a given point'''
    return 2 * np.abs(R(r, z, l, p, wz, k, zr)**2) * (1 + np.cos(2*l*phi))

def Ulp(r, z, l, p, wz, k, zr, phi):
    return np.sqrt( (2*np.math.factorial(p)) / (np.pi*np.math.factorial(p + np.abs(l))) ) * (1/wz) * (r*np.sqrt(2)/wz) ** np.abs(l) \
        * np.exp(-(r/wz)**2) * np.exp(-0.5j*k*z*r**2/(z**2 + zr**2)) * np.exp(-1j*l*phi) * np.exp(1j*(np.abs(l)+2*p+1)*np.arctan(z/zr))

def intensityGenU(r, z, l, p, wz, k, zr, phi):
    '''Fucntion to calculate the intensity at a given point'''
    return np.abs(Ulp(r, z, l, p, wz, k, zr, phi)+Ulp(r, z, -l, p, wz, k, zr, phi))**2

def distances(N, sca):
    distance = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            x = (i - (N / 2) + 0.5)
            y = (j - (N / 2) - 0.5)
            distance[i,j] = np.sqrt(x**2 + y**2) * sca
    return distance

def angles(N):
    angle = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            x = (i - (N / 2) + 0.5)
            y = (j - (N / 2) - 0.5)
            angle[i,j] = np.arctan(x / y)
    return angle
       
def CMgraph(Data):
    plt.figure()
    im = plt.imshow( Data, cmap = cm.viridis)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.show()

def CMgraphBar(Data, Label):
    plt.figure()
    im = plt.imshow( Data, cmap = cm.viridis)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.colorbar(im, orientation = 'vertical', label = Label)
    plt.show()