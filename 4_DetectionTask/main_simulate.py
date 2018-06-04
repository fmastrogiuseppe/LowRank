
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 4: sample rank-one network for the Go-Nogo detection (related to Fig. 4 B-C-D-E)
#### The code is composed of three main scripts, to be run in the order:
#### main.simulate.py  -  generates a network and the activity in response to four stimulus intensities,
####                      for Nic different noise realizations
#### main.plot.py  -  plots single neuron and population activity
#### main.regression.py  -  computes the regression axis for input and choice


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import matplotlib.pyplot as plt
import numpy as np

import fct_simulations as sim
import fct_facilities as fac


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters

# Network

g = 0.8      # Random strength

Sii = 1.2     # Std of input stimuli
Siw = 1.2     # Std of readout
rhom = 2.     # Connectivity overlap for m
rhon = 2.     # Connectivity overlap for n
ParVec = [ Sii, Siw, rhom, rhon ]

# Simulation

N = 2500     # Number of units

Nic = 12      # Number of realizations of the random input
sigma = 0.4  # Std of noise in input

T0 = 10      # Time of integration: initial transient to discard
T1 = 7       # Time of integration: resting state
T2 = 22      # Time of integration: stimulus presentation

deltat = 0.1

t0 = np.linspace( 0, T0, int(T0/deltat) )
t1 = np.linspace( 0, T1, int(T1/deltat) )
t2 = np.linspace( 0, T2, int(T2/deltat) )

t = np.concatenate([ t1, T1+t2 ])

path_data = 'Data/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Simulation
# Simulate activity in response to inputs of four different strengths

c_values = np.array([ 0.2, 0.4, 0.6, 0.9 ])

readout = np.zeros (( len(c_values),  Nic, len(t1) + len(t2) ))        # Readout z
activation = np.zeros (( len(c_values), Nic, len(t1) + len(t2), N ))   # Activarion variable x
structure = np.zeros (( 4, N ))   # Save the vectors which define the task:

# Generate the task vectors

w = sim.GetGaussianVector( 0, Siw, N )     # Readout
I = sim.GetGaussianVector( 0, Sii, N )     # Stimulus direction

y = sim.GetGaussianVector( 0, 1, N )     # Direction of connectivity overlap (see Methods)
h = sim.GetGaussianVector( 0, 1, N )     # Orthogonal noise in input (see Methods)

structure[0,:] = w
structure[1,:] = I
structure[2,:] = y
structure[3,:] = h

# Generate the matrix

m = w + rhom*y
n = I + rhon*y

M = np.outer( m, n ) / N
R = g * sim.GetBulk ( N )
J = R + M

#### Phase 0: initialize dynamics on negative solution, discard the transient

Z = sim.SimulateActivity_Noise ( t0, - m - n, J, c = 0, sigma = 0, I = 0, h = 0)

#### Phase 1: no inputs

Z_start = sim.SimulateActivity_Noise ( t1, Z[-1,:], J, c = 0, sigma = 0, I = 0, h = 0)

#### Phase 2: input

for i, c in enumerate(c_values):

	print '----', c

	for j in range(Nic):

		print j

		activation[i, j, 0:len(t1), :] = Z_start
		readout[i, j, 0:len(t1)] = np.dot(np.tanh(Z_start), w) / N

		Z = sim.SimulateActivity_Noise ( t2, Z_start[-1,:], J, c = c, sigma = sigma, I = I, h = h)

		activation[i, j, len(t1):, :] = Z
		readout[i, j, len(t1):] = np.dot(np.tanh(Z), w) / N

# Store

fac.Store(ParVec, 'ParVec.p', path_data)
fac.Store(c_values, 'c_values.p', path_data)
fac.Store(sigma, 'sigma.p', path_data)

fac.Store(readout, 'readout.p', path_data)
fac.Store(activation, 'activation.p', path_data)
fac.Store(structure, 'structure.p', path_data)

fac.Store(t, 't.p', path_data)
fac.Store(t1, 't1.p', path_data)
fac.Store(t2, 't2.p', path_data)

#
