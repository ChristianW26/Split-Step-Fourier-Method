# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 15:40:38 2025

@author: CKW09
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt

#%% Parameters
T0 = 1e-12              # s
B2 = -20e-27            # s^2/m
n2 = 3e-20              # m^2/W
Aeff = 50e-12           # m^2
c = 3e8                 # m/s
w0 = 2e14               # rad/s
P0 = 50e3               # W

#%% Determine lengths 
gamma = n2*w0/(c*Aeff)
L_D = T0**2/abs(B2)     # m
L_NL = 1/(gamma*P0)

print(f'L_D = {L_D:.2f} m')
print(f'L_NL = {L_NL:.2f} m')
print(f'L_D/L_NL = {L_D/L_NL:.2e}')    

#%% Split Step Fourier Parameters
# Number of samples
max_z = 3*L_NL
N_z = 3000
h = max_z/N_z

N_tau = 1024
tau_max = 20
tau = np.linspace(-1*tau_max, tau_max, N_tau, endpoint=False)
f = fftfreq(N_tau, tau[1]-tau[0])

U = np.zeros((N_z+1, N_tau), dtype=complex)

# Initial function U(0, Tau)
U[0] = np.exp(-.5*tau**2)
# U[0] = 1/np.cosh(tau)

#%% Helper functions
def sign(x):
    if x==0: 
        return 0
    elif x>0:
        return 1
    else:
        return -1

def dispersion_evolution(U, h):
    U_f = fft(U)
    D_hat = -1j*sign(B2)*(2*np.pi*f)**2/(2*L_D)
    res_f = np.exp(h*D_hat)*U_f
    return ifft(res_f)

def spm_evolution(U, h):
    N_hat = 1j*np.abs(U)**2/L_NL
    return np.exp(h*N_hat)*U

#%% Main Loop
for i in range(1, N_z+1):
    temp = U[i-1]                                       # Get last time-step
    temp = dispersion_evolution(temp, h/2)         # Apply first half of dispersion
    temp = spm_evolution(temp, h)                  # Apply SPM
    U[i] = dispersion_evolution(temp, h/2)         # Apply second half of dispersion
    
#%% Plotting
# Time domain
fig = plt.figure(num=1, clear=True, figsize=(15,5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(tau, np.abs(U[0])**2, 'k-', label = '$z=0$')
ax.plot(tau, np.abs(U[N_z//3])**2, 'b-', label = '$z=L$')
ax.plot(tau, np.abs(U[2*N_z//3])**2, 'r-', label = '$z=2L$')
ax.plot(tau, np.abs(U[N_z])**2, 'g-', label = '$z=3L$')
ax.grid(True)
ax.legend(loc='best')
ax.set_ylabel(r'$|U|^2$', size=18)
ax.set_xlabel(r'$\tau$', size=18)
#ax.set_xlim(-10, 10) 

# Freq domain (absolute value)
f_plot = fftshift(f)
U_0_f_plot = 1.0/N_tau*np.abs(fftshift(fft(U[0])))
U_L_f_plot = 1.0/N_tau*np.abs(fftshift(fft(U[N_z//3])))
U_2L_f_plot = 1.0/N_tau*np.abs(fftshift(fft(U[2*N_z//3])))
U_3L_f_plot = 1.0/N_tau*np.abs(fftshift(fft(U[N_z])))

ax = fig.add_subplot(1, 2, 2)
ax.plot(f_plot, U_0_f_plot, 'k-', label = '$z=0$')
ax.plot(f_plot, U_L_f_plot, 'b-', label = '$z=L$')
ax.plot(f_plot, U_2L_f_plot, 'r-', label = '$z=2L$')
ax.plot(f_plot, U_3L_f_plot, 'g-', label = '$z=3L$')
ax.grid(True)
ax.legend(loc='best')
ax.set_ylabel(r'$|S(f)|$', size=18)
ax.set_xlabel(r'$f$', size=18) 
ax.set_xlim(-5, 5)

fig.tight_layout()