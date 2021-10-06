# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:18:44 2018

@author: Rene
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import exp
from numpy.fft import rfft2,irfft2

#LOADING THE BLURRED IMAGE
b = np.loadtxt("blur.txt", ndmin=2)

#GENERATING THE POINT SPREAD FUNCTION (PSF)

L = len(b)
Gaussian = np.zeros(shape = (L,L), dtype = float)
x = np.arange(L)
y = np.arange(L)

sigma = 25

f = lambda x,y: exp(-(x*x+y*y)/(2*sigma*sigma))
for i in range(L):
	for j in range(L):
		Gaussian[i,j]+=f(i,j)
		Gaussian[-i,-j]+=f(i,j)
		Gaussian[-i,j]+=f(i,j)
		Gaussian[i,-j]+=f(i,j)
        
#FOURIER TRANSFORM OF THE BLURRED PHOTO AND THE PSF
b_ik = rfft2(b)
f_ik = rfft2(Gaussian)

#DIVIDING BOTH TO OBTAIN THE FOURIER TRANSFORM OF THE UNBLURRED PHOTO

limit = 1e-3
a_ik = b_ik/f_ik
a_ik[f_ik>limit] = b_ik[f_ik>limit]/f_ik[f_ik>limit]
a_ik[f_ik<=limit] = b_ik[f_ik<=limit]

#PERFORMING INVERSE TRANSFORM TO GET THE UNBLURRED PHOTO
a = irfft2(a_ik)

#PLOTS 
plt.title("Blurred Photo")
plt.imshow(b)
plt.gray()
plt.show()
plt.title("Point Spread Function")
plt.imshow(Gaussian)
plt.gray()
plt.show()
plt.title("Deblurring Attempt using Deconvolution")
plt.imshow(a)
plt.gray()
plt.show()

print("There are two problems that arise, one is that convolution does not exactly (100%) model blurring of the image in the camera,")
print("and another one is inverse filtering is said to be numerically unstable (that's why limit is set).")
print("Estimating the PSF is very crucial, and very slight innacuracy could be greatly amplified in a phenomenon called 'ringing', which makes the method very unstable.")
print("")
print("Source: Scholnik, S. 'Digital Signal Processing: Do you know the reasons why image deconvolution (deblur) does not always work?'. Quora. (2014)")
print("URL:https://www.quora.com/Digital-Signal-Processing-Do-you-know-the-reasons-why-image-deconvolution-deblur-does-not-always-work")
print("Date retrieved: April 7, 2018")