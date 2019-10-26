from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from math import pi, gamma, cos, sin


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
dk = 5e-3
thetar = 0.0
cen = np.floor(N/2)


#################################################


        #Generate phase screens
        #potentially change generation to be 1 screen/1 km
b = 1.0
c = 1.0
thetar = (np.pi/180.0)*0.0
delta =dk #Spatial sampling rate
        
del_f = 1.0/(N * delta) #Frequency grid spacing(1/m)
cen = np.floor(N/2)
        
na = 22.0/6.0 #Normalized alpha value
Bnum = gamma(na/2.0)
Bdenom = 2.0**(2.0-na)*pi*na*gamma(-na/2.0)
        
        #c1 Striblings Consistency parameter. Evaluates to 6.88 in Kolmogorov turb.
cone = (2.0* (8.0/(na-2.0) *gamma(2.0/(na-2.0)))**((na-2.0)/2.0))
        
        #Charnotskii/Bos generalized phase consistency parameter
Bfac = (2.0*pi)**(2.0-na) * (Bnum/Bdenom)
a = gamma(na-1.0)*cos(na*pi/2.0)/(4.0*pi**2.0)
        # Toselli's inner-scale intertial range consistency parameter 
c_a = (gamma(0.5*(5.0-na))*a*2.0*pi/3.0)**(1.0/(na-5.0))

fm = c_a/l0   # Inner scale frequency(1/m) 
        
        # Set up parameters for Kolmogorov PSD
nae = 22/6.0 #Normalized alpha value
Bnume = gamma(nae/2.0)
Bdenome = 2.0**(2.0-nae)*pi*nae*gamma(-nae/2.0)
conee = (2.0* (8.0/(nae-2.0) *gamma(2.0/(nae-2.0)))**((nae-2.0)/2.0))
Bface = (2.0*pi)**(2.0-nae) * (Bnume/Bdenome)
ae = gamma(nae-1.0)*cos(nae*pi/2.0)/(4.0*pi**2.0)
c_ae = (gamma(0.5*(5.0-nae))*ae*2.0*pi/3.0)**(1.0/(nae-5.0))
fme = c_ae/l0   # Inner scale frequency(1/m)         
        
f0 = 1.0/L0   # Outer scale frequency
        # Create frequency sample grid
fx = np.arange(-N/2.0, N/2.0) * del_f
fx, fy = np.meshgrid(fx,-1*fx)
        
        # Apply affine transform
tx = fx*cos(thetar) + fy*sin(thetar)
ty = -1.0*fx*sin(thetar) + fy*cos(thetar)
        
        # Scalar frequency grid 
f = np.sqrt((tx**2.0)/(b**2.0) + (ty**2.0)/(c**2.0))

        # Sample Turbulence PSD
PSD_phi = (cone * Bfac * ((b*c)**(-na/2.0)) * (45.0**(2.0-na)) * np.exp(-(f/fm)**2.0) \
/((f**2.0 + f0**2.0)**(na/2.0)))
        
tot_NOK = np.sum(PSD_phi)
        
        # Kolmogorov equivalent and enforce isotropy
        # Sample Turbulence PSD
PSD_phie = (conee * Bface * (45.0**(2.0-nae)) * np.exp(-(f/fme)**2.0) \
/((f**2.0 + f0**2.0)**(nae/2.0)))
        
tot_OK = np.sum(PSD_phie)
        
PSD_phi = (tot_OK/tot_NOK) * PSD_phi
   
        #PSD_phi = cone*Bfac* (r0**(2-na)) * f**(-na/2)  # Kolmogorov PSD
#PSD_phi[np.int(cen),np.int(cen)]=0.0

        # Create a random field that is circular complex Guassian
cn = (np.random.randn(N,N) + 1j*np.random.randn(N,N) )

        # Filter by turbulence PSD
cn = cn * np.sqrt(PSD_phi)*del_f

        # Inverse FFT
phz_temp  = np.fft.ifft2(np.fft.fftshift(cn))*((N)**2)

        # Phase screens 
phz1 = np.real(phz_temp)


#################################################


plt.figure()
im = plt.imshow( phz1, cmap = cm.viridis)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.show()