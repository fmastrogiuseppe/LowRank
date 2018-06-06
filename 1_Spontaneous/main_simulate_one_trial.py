
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 1b: spontaneous activity in rank-one networks - simulate a finite-size network (related to Fig. 1B)

#### Note that the Data/ folder is empty to begin; this code needs to be run with the flag doCompute = 1
#### at least once


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import numpy as np
import matplotlib.pyplot as plt

import fct_simulations as sim
import fct_facilities as fac


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters

# Network

Mm = 1.1        # Mean of m
Mn = 2.         # Mean of n
Sim = 1.        # Std of m
Sin = 1.        # Std of n

g = 1.          # Random strength

# Simulation

N = 1000       # Number of units

T = 40         # Total time of integration, expressed in time constants of single units
deltat = 0.1
t = np.linspace( 0, T, int(T/deltat) )

Nsample = 5    # Store the activity of Nsample units


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute

doCompute = 1
path_here = 'Data/'

if doCompute == 1:

	### Generate the matrix

	R = g * sim.GetBulk ( N )
	m = sim.GetGaussianVector( Mm, Sim, N )
	n =  sim.GetGaussianVector( Mn, Sin, N )
	M = np.outer( m, n ) / N
	J = R + M

	### Simulate

	Z = sim.SimulateActivity ( t, sim.GetGaussianVector(0, 1, N), J, I=0 )

	# Store

	fac.Store(Z[:,0:Nsample], 'Z.p', path_here)

else:

	# Retrieve

	Z = fac.Retrieve('Z.p', path_here)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot

fac.SetPlotParams()

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for i in range(Nsample):
	plt.plot(t, (Z[:, i]), color = '0.6')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.xlabel('time (norm.)')
plt.ylabel('Activation $x_i$')
plt.savefig('trial.pdf')
plt.show()

#
