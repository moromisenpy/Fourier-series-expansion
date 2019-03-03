#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 02:21:33 2019

@author: moromi_senpy
"""
#############################################################
#               Fourier series expansion
#############################################################

#============================================================
#                     Import modules
#============================================================
import numpy as np
import matplotlib.pyplot as plt
import math
#============================================================
#                Definition of Function
#============================================================

# f(t) = Pi(x) : rectangular function
def f(t, L):
    Pi = np.zeros_like(t)
    index = np.abs(t) < (L/2)
    Pi[index] = 1
    return Pi

#============================================================
#                     Main Routine
#============================================================

tap = 1000
t = np.linspace(-np.pi, np.pi, num=tap)
f_hat = np.zeros_like(t)
L = np.pi
N = 50 # order
a = np.empty([N+1], dtype=np.float64) # coefficients
b = np.empty([N+1], dtype=np.float64) # coefficients

# calculate coefficients
a[0] = np.sum(f(t,L)) / tap
for n in range(1, N+1):
    a[n] = np.sum(f(t,L)*np.cos(n*t)) / (tap/2)
    b[n] = np.sum(f(t,L)*np.sin(n*t)) / (tap/2)

# draw function
fig = plt.figure()
plt.plot(t, f(t,L), linewidth=4, color="red", label=r"$\Pi(\frac{L}{2})$")
for n in range(N+1):
    f_hat = f_hat + a[n]*np.cos(n*t) + b[n]*np.sin(n*t)
    if n%10==0:
        plt.plot(t, f_hat, linestyle="dashed", label=r"$a_{0} + \sum_{n=1}^{%d} (a_{n} \cos nt + b_{n} \sin nt)$" %(n))
    
plt.title("Fourier series expansion")
plt.xlabel("$t$")
plt.xlim([-np.pi,np.pi])
plt.ylim([-0.3,1.3])
plt.grid(True)
plt.legend()
fig.tight_layout()
