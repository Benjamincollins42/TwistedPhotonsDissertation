from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


L = 3 #Length
N = 512 #Number of pixels
l0 = 0.01
L0 = 50
wvl = 500e-9
wvn = 2*np.pi / wvl
dz = 100
fm = 5.92/l0
f0 = 2*np.pi / L0
cn2 = 1e-16
dx = L / N #Works when number is fudged but
df = 1 / (N*dx)


#Scalar frequency grid
fx = np.arange(-N/2.0, N/2.0) * df
fx, fy = np.meshgrid(fx, -1*fx)
f = np.sqrt( (fx**2) + (fy**2) )


#Power Density
PDS = 0.033*cn2*np.exp( - (f/fm)**2  ) / np.power( (f**2 + f0**2), (11/3), )
#PDS[N/2,N/2] = 0.0

#Random array generation
cn = np.random.randn(N,N) + 1j*np.random.randn(N,N)


#Thing to inverse
cn = cn * np.sqrt(PDS*dz*2*np.pi) * (2*np.pi*wvn / (N*df) )


phz_tmp = np.fft.ifft2(np.fft.ifftshift(cn))
result = np.real(phz_tmp)


plt.figure()
im = plt.imshow( result, cmap = cm.viridis)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.show()