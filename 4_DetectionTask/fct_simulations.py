import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import pickle
from functools import partial

### Functions for building CONNECTIVITY MATRIX

def GetBulk (N):
    chi = np.random.normal( 0, np.sqrt(1./(N)), (N,N) )
    return chi

def GetGaussianVector (mean, std, N):

	if std>0:
		return np.random.normal (mean, std, N )
	else:
		return mean*np.ones(N)

### Functions for INTEGRATING

def SimulateActivity_Noise (t, x0, Jeff, c, sigma, I, h):

    Z = np.zeros(( len(t), Jeff.shape[0] ))
    Z[0,:] = x0

    deltat = t[-1] / len(t)

    for i in range(len(t) - 1):
        Z[i+1,:] = Z[i,:] + deltat * (- Z[i,:] + np.dot( Jeff, np.tanh(Z[i,:])) + c * I + h ) + np.sqrt(deltat) *  sigma * GetGaussianVector( 0, 1, 1 ) * I

    return Z