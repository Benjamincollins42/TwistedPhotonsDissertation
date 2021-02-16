'''A module containing functions for generating the intensity of a Laguerre-Gauss beam.'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Functions describing the beam


#Functions generating vaules that don't change over location
def zR(w0, La):
    '''Function to calculate the Rayleigh range.

    Args:
    w0 = beam waist
    La = (lambda) wavelength
    '''
    return np.pi * w0 ** 2 / La


def wZ(w0, z, zR):
    '''Function to calculate the spot size parameter.

    Args:
    w0 = beam waist
    z = the z axis coordinate
    zR = Rayleigh range
    '''
    return w0 * np.sqrt(1 + (z / zR) ** 2)


def k(La):
    '''Function to calculate the wavenumber.

    Args:
    La = (lambda) wavelength
    '''
    return 2 * np.pi / La


#Functions generating vaules that change over location

def R(r, z, l, p, wZ, k, zR):
    '''Function to calculate the radial part of a beam.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    '''
    return np.sqrt( (2*np.math.factorial(p)) / (np.pi*np.math.factorial(p + np.abs(l))) ) * (1/wZ) * (r*np.sqrt(2)/wZ) \
           ** np.abs(l) * np.exp(-(r/wZ)**2) * np.exp(-0.5j*k*z*r**2/(z**2 + zR**2))


def intensityGenR(r, z, l, p, wZ, k, zR, phi):
    '''Function to calculate the intensity at a given point given the radial part of the beam.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    phi = phase
    '''
    return 2 * np.abs(R(r, z, l, p, wZ, k, zR)**2) * (1 + np.cos(2*l*phi))


#Functions to put it all together into final beam intensities

def Ulp(r, z, l, p, wZ, k, zR, phi):
    '''Fucntion to calculate the complex value of the wave equation at a given point.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    phi = phase
    '''
    return np.sqrt( (2*np.math.factorial(p)) / (np.pi*np.math.factorial(p + np.abs(l))) ) * (1/wZ) * (r*np.sqrt(2)/wZ) \
           ** np.abs(l) * np.exp(-(r/wZ)**2) * np.exp(-0.5j*k*z*r**2/(z**2 + zR**2)) * np.exp(-1j*l*phi) * np.exp(1j * \
           (np.abs(l)+2*p+1)*np.arctan(z/zR))


def intensityGenU(r, z, l, p, wz, k, zr, phi):
    '''Fucntion to calculate the intensity at a given point of the petal mode superposition.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    phi = phase
    '''
    return np.abs(Ulp(r, z, l, p, wz, k, zr, phi)+Ulp(r, z, -l, p, wz, k, zr, phi))**2


def UGen(w0, La, zInitial, nPix, frameSize, l):
    '''Fucntion to calculate the intensity, at all the points of a grid, of the petal mode superposition.

    Args:
    w0 = beam waist
    La = (lambda) wavelength
    zInitial = the initial z axis coordinate
    nPix = number of pixels in a grid axis (square images only)
    frameSize = physical size of the frame
    l = quantum number for angular mode
    '''

    #Generating the required constants
    waveNumber = 2*np.pi/La
    rayleighRange0 = zR(w0, La)
    spotSizeParam0 = wZ(w0, zInitial, rayleighRange0)

    #Setting up the grids
    ui = np.zeros((nPix,nPix), dtype = 'complex128')
    Dist = distances(nPix, frameSize/nPix)
    Angl = angles(nPix)

    #Generating loop
    for i in range(0, nPix):
        for j in range(0, nPix):
            ui[i,j] = Ulp(Dist[i,j], zInitial,  l, 0, spotSizeParam0, waveNumber, rayleighRange0, Angl[i,j])
    return ui


def waveMerge(u1, u2):
    '''Function to merge the complex values of 2 waves to get the resultant intensities.

    Args:
    u1 = the complex values of wave 1
    u2 = the complex values of wave 2
    '''
    I = np.abs(u1 + u2) ** 2
    return I


#Related utillity functions

def distances(N, scale):
    '''Function to generate a grid of radial distances from the grid centre.

    Args:
    N = number of pixels in the grid (must be even)
    scale = physical size of the grid
    '''
    distance = np.zeros((N,N))

    #Offsetting the values and calculating distance
    for i in range(0, N):
        for j in range(0, N):
            x = (i - (N / 2) + 0.5)
            y = (j - (N / 2) - 0.5)
            distance[i,j] = np.sqrt(x**2 + y**2) * scale

    return distance


def angles(N):
    '''Function to generate a grid of radial distances from the grid centre.

    Args:
    N = number of pixels in the grid (must be even)
    '''
    angle = np.zeros((N,N))

    #Offsetting the values and calculating angle
    for i in range(0, N):
        for j in range(0, N):
            x = (i - (N / 2) + 0.5)
            y = (j - (N / 2) - 0.5)
            angle[i,j] = np.arctan2(y, x)
    return angle


def CMgraph(Data):
    '''Function to generate a colour map of an array of data.

    Args:
    Data = array of floats
    '''
    plt.figure()
    im = plt.imshow( Data, cmap = cm.viridis)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.show()


def CMgraphBar(Data, Label):
    '''Function to generate a colour map of an array of data, with a colour scale.

    Args:
    Data = array of floats
    Label = string describing the values in the array
    '''
    plt.figure()
    im = plt.imshow( Data, cmap = cm.viridis)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.colorbar(im, orientation = 'vertical', label = Label)
    plt.show()
