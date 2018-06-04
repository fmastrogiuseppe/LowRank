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

def Integrate (X, t, J, I):
    dXdT = -X + np.dot( J, np.tanh(X) ) + I
    return dXdT

def SimulateActivity (t, x0, Jeff, I):
    print ' ** Simulating... **'
    return scipy.integrate.odeint( partial(Integrate, J=Jeff, I=I), x0, t )
