from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

#Variables


N = 512
cen = np.floor(N/2.0)
scale = 8 / N


#Functions


def gaussianGeneration(A = 1.0, sigmaX = 1.0, sigmaY = 1.0):
    """Generates a 2D gaussian array centred on the centre coords"""
    GaussianA = np.zeros((N,N))
    for x in range(0,N):
        for y in range(0,N):
            GaussianA[x,y] = A * np.exp( - ( ((x - cen)*scale)**2/(2*sigmaX**2) + ((y - cen)*scale)**2/(2*sigmaY**2) ) )
    return GaussianA


def fourierT(inputA, theta = np.zeros((N,N))):
    return np.fft.fft2(inputA)


def thereAndBack(inputA):
    return np.fft.ifft2(np.fft.fft2(inputA))


G = gaussianGeneration()
#Graph Plotting
plot = (np.real(thereAndBack(G)) - G)/G
plt.figure()
im = plt.imshow( plot, cmap = cm.viridis)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.show()

J = np.amax(plot)*100
print "Max percentage error: "
print J