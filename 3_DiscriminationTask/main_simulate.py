
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 3: sample rank-one network for the Go-Nogo discrimination task (related to Fig. 3 B-C-E-F)
#### The code is composed of three main scripts, to be run in the order:
#### main_simulate.py  -  generates a network and the activity in response to Go and Nogo stimulus
#### main_plot.py  -  plots single neuron and population activity
#### main_PCA_connectivity.py  -  computes the first PC axis for Go and Nogo trials, and correlates
####                              them with the average pair connectivity

#### Note that the Data/ folder is empty to begin; the code main_simulate.py needs to be run  
#### at least once


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import matplotlib.pyplot as plt
import numpy as np

import fct_simulations as sim
import fct_facilities as fac


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters

# Network

g = 0.1      # Random strength

Sii = 2.     # Std of input stimuli
Siw = 2.     # Std of readout
ParVec = [ Sii, Siw ]

# Simulation

N = 2500     # Number of units

T0 = 10      # Time of integration: initial transient to discard
T1 = 5       # Time of integration: resting state
T2 = 10      # Time of integration: stimulus presentation
T3 = 15      # Time of integration: decay to rest

deltat = 0.1

t0 = np.linspace( 0, T0, int(T0/deltat) )
t1 = np.linspace( 0, T1, int(T1/deltat) )
t2 = np.linspace( 0, T2, int(T2/deltat) )
t3 = np.linspace( 0, T3, int(T3/deltat) )

t = np.concatenate([ t1, T1+t2, T1+T2+t3 ])

path_data = 'Data/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Simulation
# Simulate activity in response the Go and the Nogo stimulus

readout = np.zeros (( 2,  len(t1) + len(t2) + len(t3) ))         # Readout z
activation = np.zeros (( 2,  len(t1) + len(t2) + len(t3), N ))   # Activation variable x
structure = np.zeros (( 3, N ))    # Save the vectors which define the task: w, I0 and I1

# Generate the task vectors

w = sim.GetGaussianVector( 0, Siw, N )     # Readout
IA = sim.GetGaussianVector( 0, Sii, N )    # Go stimulus
IB = sim.GetGaussianVector( 0, Sii, N )    # Nogo stimulus

structure[0,:] = w
structure[1,:] = IA
structure[2,:] = IB

# Generate the matrix

M = np.outer( w, IA ) / N    # Take the connectivity vectors m and n to be equal to w and IA
R = g * sim.GetBulk ( N )
J = R + M

average_connectivity = 0.5 * (J + J.T)

#### Phase 0: simulate and discard the transient

Z = sim.SimulateActivity ( t0, sim.GetGaussianVector( 0, 0.1, N ), J, I = 0 )

#### Phase 1: no inputs

Z = sim.SimulateActivity ( t1, Z[-1,:], J, I = 0 )

activation[0, 0:len(t1), :] = Z
activation[1, 0:len(t1), :] = Z

readout[0, 0:len(t1)] = np.dot(np.tanh(Z), w) / N
readout[1, 0:len(t1)] = np.dot(np.tanh(Z), w) / N

#### Phase 2: Go and Nogo inputs

Z0 = sim.SimulateActivity ( t2, Z[-1,:], J, IA )
Z1 = sim.SimulateActivity ( t2, Z[-1,:], J, IB )

activation[0, len(t1):len(t1)+len(t2), :] = Z0
activation[1, len(t1):len(t1)+len(t2), :] = Z1

readout[0, len(t1):len(t1)+len(t2)] = np.dot(np.tanh(Z0), w) / N
readout[1, len(t1):len(t1)+len(t2)] = np.dot(np.tanh(Z1), w) / N

### Phase 3: no inputs

Z0 = sim.SimulateActivity ( t3, Z0[-1,:], J, I = 0 )
Z1 = sim.SimulateActivity ( t3, Z1[-1,:], J, I = 0 )

activation[0, len(t1)+len(t2):, :] = Z0
activation[1, len(t1)+len(t2):, :] = Z1

readout[0, len(t1)+len(t2):] = np.dot(np.tanh(Z0), w) / N
readout[1, len(t1)+len(t2):] = np.dot(np.tanh(Z1), w) / N

# Store

fac.Store(ParVec, 'ParVec.p', path_data)

fac.Store(readout, 'readout.p', path_data)
fac.Store(activation, 'activation.p', path_data)
fac.Store(structure, 'structure.p', path_data)

fac.Store(t, 't.p', path_data)
fac.Store(t1, 't1.p', path_data)
fac.Store(t2, 't2.p', path_data)
fac.Store(t3, 't3.p', path_data)

fac.Store(average_connectivity, 'average_connectivity.p', path_data)

#
