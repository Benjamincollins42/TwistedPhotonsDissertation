"""A module containing functions for generating the intensity of a Laguerre-Gauss beam."""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import phaseScreenGeneration as PS

# Functions describing the beam

class LGBeam:
    """
    A class to contain a Laguerre-Gauss beam and allow for its propagation through phasescreens.
    """

    def __init__(self, l=1, p=0, w0=0.13, La=5e-7, z0=0, N=256, L=3):
        self.l = l
        self.p = p
        self.w0 = w0
        self.La = La
        self.z0 = z0
        self.N = N
        self.L = L
        self.dx = self.L / self.N
        self.zR = self.zR()
        self.k = self.k()
        self.r = self.distances()
        self.phi = self.angles()
        self.ui = self.UGen(self.z0)
        self.u = self.ui
        self.xf, self.yf = self.freqRanges()

    def zR(self):
        """Function to calculate the Rayleigh range.

        Args:
        w0 = beam waist
        La = (lambda) wavelength
        """
        return np.pi * self.w0 ** 2 / self.La

    def k(self):
        """Function to calculate the wavenumber.

        Args:
        La = (lambda) wavelength
        """
        return 2 * np.pi / self.La

    def wZ(self, z):
        """Function to calculate the spot size parameter.

        Args:
        w0 = beam waist
        z = the z axis coordinate
        zR = Rayleigh range
        """
        return self.w0 * np.sqrt(1 + (z / self.zR) ** 2)

    def R(self, r, z):
        """Function to calculate the radial part of a beam.

        Args:
        r = radial distance from beam centre
        z = the z axis coordinate
        l = quantum number for angular mode
        p = quantum number for radial mode
        wZ = spot size parameter
        k = wavenumber
        zR = Rayleigh range
        """
        wZ = self.wZ(z)
        return np.sqrt((2 * np.math.factorial(self.p)) / (np.pi * np.math.factorial(self.p + np.abs(self.l)))) * (1 / wZ
                                                                                                                  ) * (
                           r * np.sqrt(2) / wZ) ** np.abs(self.l) * np.exp(-(r / wZ) ** 2) * np.exp(
            -0.5j * self.k * z * self.r **
            2 / (z ** 2 + self.zR ** 2))

    def intensityGenR(self, r, z, phi):
        """Function to calculate the intensity at a given point given the radial part of the beam.

        Args:
        r = radial distance from beam centre
        z = the z axis coordinate
        l = quantum number for angular mode
        p = quantum number for radial mode
        wZ = spot size parameter
        k = wavenumber
        zR = Rayleigh range
        phi = phase
        """
        return 2 * np.abs(self.R(r, z) ** 2) * (1 + np.cos(2 * self.l * phi))

    def Ulp(self, r, z, phi):
        """Function to calculate the complex value of the wave equation at a given point.

        Args:
        r = radial distance from beam centre
        z = the z axis coordinate
        l = quantum number for angular mode
        p = quantum number for radial mode
        wZ = spot size parameter
        k = wavenumber
        zR = Rayleigh range
        phi = phase
        """
        wZ = self.wZ(z)
        return np.sqrt((2 * np.math.factorial(self.p)) / (np.pi * np.math.factorial(self.p + np.abs(self.l)))) * (1 / wZ
                                                                                                                  ) * (
                           r * np.sqrt(2) / wZ) ** np.abs(self.l) * np.exp(-(r / wZ) ** 2) * np.exp(
            -0.5j * self.k * z * r ** 2 /
            (z ** 2 + self.zR ** 2)) * np.exp(-1j * self.l * phi) * np.exp(1j * (np.abs(self.l) + 2 * self.p + 1) *
                                                                           np.arctan(z / self.zR))

    def distances(self):
        """Function to generate a grid of radial distances from the grid centre.

        Args:
        N = number of pixels in the grid
        L = physical size of the grid
        """
        distance = np.zeros((self.N, self.N))

        # Offsetting the values and calculating distance

        for i in range(0, self.N):
            for j in range(0, self.N):
                x = (i - (self.N / 2) + 0.5)
                y = (j - (self.N / 2) - 0.5)
                distance[i, j] = np.sqrt(x ** 2 + y ** 2) * self.dx

        return distance

    def angles(self):
        """Function to generate a grid of radial distances from the grid centre.

        Args:
        N = number of pixels in the grid
        """
        angle = np.zeros((self.N, self.N))

        # Offsetting the values and calculating angle

        for i in range(0, self.N):
            for j in range(0, self.N):
                x = (i - (self.N / 2) + 0.5)
                y = (j - (self.N / 2) - 0.5)
                angle[i, j] = np.arctan2(y, x)
        return angle

    def UGen(self, z):
        """Function to calculate the intensity, at all the points of a grid, of the petal mode superposition.

        Args:
        w0 = beam waist
        La = (lambda) wavelength
        zInitial = the initial z axis coordinate
        nPix = number of pixels in a grid axis (square images only)
        frameSize = physical size of the frame
        l = quantum number for angular mode
        """

        # Generating the required constants

        # Setting up the grid
        ui = np.zeros((self.N, self.N), dtype='complex128')

        # Generating loop
        for i in range(0, self.N):
            for j in range(0, self.N):
                ui[i, j] = self.Ulp(self.r[i, j], z, self.phi[i, j])
        return ui

    def propagate(self, dz, screen):
        """
        FILL HERE

        Args:
        dz = distance propagated
        U0 = input wave
        N = pixel width
        L = length of grid side
        screen = phasescreen
        wnn = wavenumber
        """
        Um = np.fft.fftshift(np.fft.fftn(self.u * np.exp(1j * screen)) * self.dx ** 2)
        for i in range(self.N):
            for j in range(self.N):
                Um[i, j] = np.exp(1j * (self.k ** 2 - (self.xf[j] * 2 * np.pi) ** 2 - (self.yf[i] * 2 * np.pi) ** 2) ** 0.5 * dz) * Um[i, j]

        Um = np.fft.ifftn(np.fft.fftshift(Um)) / self.dx ** 2

        self.u = Um

    def freqRanges(self):
        '''Calculates the N xN frequency ranges of the axes.

        Args:
        N = pixel width
        L = length of grid side
        '''


        xf = np.fft.fftfreq(self.N, self.dx)
        xf = np.fft.fftshift(xf)  # Shift gets them in the right order for the fourier transforms
        yf = np.fft.fftfreq(self.N, self.dx)
        yf = np.fft.fftshift(yf)
        return xf, yf


# Functions generating values that don't change over location
def zR(w0, La):
    """Function to calculate the Rayleigh range.

    Args:
    w0 = beam waist
    La = (lambda) wavelength
    """
    return np.pi * w0 ** 2 / La


def wZ(w0, z, zR):
    """Function to calculate the spot size parameter.

    Args:
    w0 = beam waist
    z = the z axis coordinate
    zR = Rayleigh range
    """
    return w0 * np.sqrt(1 + (z / zR) ** 2)


def k(La):
    """Function to calculate the wavenumber.

    Args:
    La = (lambda) wavelength
    """
    return 2 * np.pi / La


# Functions generating values that change over location

def R(r, z, l, p, wZ, k, zR):
    """Function to calculate the radial part of a beam.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    """
    return np.sqrt((2 * np.math.factorial(p)) / (np.pi * np.math.factorial(p + np.abs(l)))) * (1 / wZ) * (
                r * np.sqrt(2) / wZ) \
           ** np.abs(l) * np.exp(-(r / wZ) ** 2) * np.exp(-0.5j * k * z * r ** 2 / (z ** 2 + zR ** 2))


def intensityGenR(r, z, l, p, wZ, k, zR, phi):
    """Function to calculate the intensity at a given point given the radial part of the beam.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    phi = phase
    """
    return 2 * np.abs(R(r, z, l, p, wZ, k, zR) ** 2) * (1 + np.cos(2 * l * phi))


# Functions to put it all together into final beam intensities

def Ulp(r, z, l, p, wZ, k, zR, phi):
    """Function to calculate the complex value of the wave equation at a given point.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    phi = phase
    """
    return np.sqrt((2 * np.math.factorial(p)) / (np.pi * np.math.factorial(p + np.abs(l)))) * (1 / wZ) * (
                r * np.sqrt(2) / wZ) \
           ** np.abs(l) * np.exp(-(r / wZ) ** 2) * np.exp(-0.5j * k * z * r ** 2 / (z ** 2 + zR ** 2)) * np.exp(
        -1j * l * phi) * np.exp(1j * \
                                (np.abs(l) + 2 * p + 1) * np.arctan(z / zR))


def intensityGenU(r, z, l, p, wz, k, zr, phi):
    """Function to calculate the intensity at a given point of the petal mode superposition.

    Args:
    r = radial distance from beam centre
    z = the z axis coordinate
    l = quantum number for angular mode
    p = quantum number for radial mode
    wZ = spot size parameter
    k = wavenumber
    zR = Rayleigh range
    phi = phase
    """
    return np.abs(Ulp(r, z, l, p, wz, k, zr, phi) + Ulp(r, z, -l, p, wz, k, zr, phi)) ** 2


def UGen(w0, La, zInitial, nPix, frameSize, l):
    """Function to calculate the intensity, at all the points of a grid, of the petal mode superposition.

    Args:
    w0 = beam waist
    La = (lambda) wavelength
    zInitial = the initial z axis coordinate
    nPix = number of pixels in a grid axis (square images only)
    frameSize = physical size of the frame
    l = quantum number for angular mode
    """

    # Generating the required constants
    waveNumber = 2 * np.pi / La
    rayleighRange0 = zR(w0, La)
    spotSizeParam0 = wZ(w0, zInitial, rayleighRange0)

    # Setting up the grids
    ui = np.zeros((nPix, nPix), dtype='complex128')
    Dist = distances(nPix, frameSize / nPix)
    Angle = angles(nPix)

    # Generating loop
    for i in range(0, nPix):
        for j in range(0, nPix):
            ui[i, j] = Ulp(Dist[i, j], zInitial, l, 0, spotSizeParam0, waveNumber, rayleighRange0, Angle[i, j])
    return ui


def waveMerge(u1, u2):
    """Function to merge the complex values of 2 waves to get the resultant intensities.

    Args:
    u1 = the complex values of wave 1
    u2 = the complex values of wave 2
    """
    I = np.abs(u1 + u2) ** 2
    return I


# Related utility functions

def distances(N, scale):
    """Function to generate a grid of radial distances from the grid centre.

    Args:
    N = number of pixels in the grid (must be even)
    scale = physical size of the grid
    """
    distance = np.zeros((N, N))

    # Offsetting the values and calculating distance
    for i in range(0, N):
        for j in range(0, N):
            x = (i - (N / 2) + 0.5)
            y = (j - (N / 2) - 0.5)
            distance[i, j] = np.sqrt(x ** 2 + y ** 2) * scale

    return distance


def angles(N):
    """Function to generate a grid of radial distances from the grid centre.

    Args:
    N = number of pixels in the grid (must be even)
    """
    angle = np.zeros((N, N))

    # Offsetting the values and calculating angle
    for i in range(0, N):
        for j in range(0, N):
            x = (i - (N / 2) + 0.5)
            y = (j - (N / 2) - 0.5)
            angle[i, j] = np.arctan2(y, x)
    return angle


def CMgraph(Data):
    """Function to generate a colour map of an array of data.

    Args:
    Data = array of floats
    """
    plt.figure()
    im = plt.imshow(Data, cmap=cm.viridis)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.show()


def CMgraphBar(Data, Label):
    """Function to generate a colour map of an array of data, with a colour scale.

    Args:
    Data = array of floats
    Label = string describing the values in the array
    """
    plt.figure()
    im = plt.imshow(Data, cmap=cm.viridis)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.colorbar(im, orientation='vertical', label=Label)
    plt.show()


if __name__ == '__main__':
    l = 16
    dz = 1000
    cn2 = 1.6e-13
    L0 = 50
    kl = 3.3/0.01
    testP = LGBeam(l=l)
    testN = LGBeam(l=-l)
    #CMgraph(waveMerge(testP.ui, testN.ui))
    test_ps = PS.phaseScreenGenInd(PS.freq(testP.N, testP.L)[0], 1/testP.dx, dz, cn2, L0, kl, testP.k, testP.N)
    CMgraph(np.real(test_ps))
    testP.propagate(100000, test_ps/testP.dx)
    testN.propagate(100000, test_ps/testN.dx)
    CMgraph(waveMerge(testP.u, testN.u)-waveMerge(testP.ui, testN.ui))
