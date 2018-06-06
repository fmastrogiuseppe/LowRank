
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 5: sample rank-two network for the contex-dependent discrimination task (related to Fig. 5 C-D-E)
#### The code is composed of two main scripts, to be run in the order:
#### main.simulate.py  -  generates a network and the activity in response to stimuli A and B
####                      in both contexts
#### main.plot.py  -  plots single neuron and population activity

#### Note that this more complex model suffers of stronger finite-size effects.
#### If the network you generated does not solve the task, you haven't been lucky, try with another one...

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

g = 0.8        # Random strength

Sii = 1.2      # Std of input stimuli

rhom = 1.6     # Internal overlap between connectivity vectors, along n
rhon = 3.      # Internal overlap between connectivity vectors, along n

betam = 0.6    # Overlap between connectivity vectors m and readout w
betan = 1.     # Overlap between connectivity vectors n and readout w

gammaON = 0.08        # Strength of the contextual input which gets activated
gammaOFF = - 0.14     # Strength of the contextual input which gets switched off

ParVec = [ Sii, rhom, rhon, betam, betan ]

# Simulation

N = 7000     # Number of units

T0 = 10      # Time of integration: initial transient to discard
T1 = 15       # Time of integration: resting state
T2 = 85      # Time of integration: stimulus presentation

deltat = 0.1

t0 = np.linspace( 0, T0, int(T0/deltat) )
t1 = np.linspace( 0, T1, int(T1/deltat) )
t2 = np.linspace( 0, T2, int(T2/deltat) )

t = np.concatenate([ t1, T1+t2 ])

path_data = 'Data/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Simulation
# Simulate activity in response to stimuli A and B for context A and B

readout_A = np.zeros (( 2,  len(t1) + len(t2) ))     # Readout z in context A
readout_B = np.zeros (( 2,  len(t1) + len(t2) ))     # Readout z in context B

activation_A = np.zeros (( 2,  len(t1) + len(t2), N ))     # Variable x in context A
activation_B = np.zeros (( 2,  len(t1) + len(t2) , N ))    # Variable x in context B

structure = np.zeros (( 11, N ))   # Save the vectors which define the task

# Generate the task vectors

IA = sim.GetGaussianVector( 0, Sii, N)      # Stimulus A
IB = sim.GetGaussianVector( 0, Sii, N)      # Stimulus B

y1 = sim.GetGaussianVector( 0, Sii, N)      # Orthogonal direction of m1
y2 = sim.GetGaussianVector( 0, Sii, N)      # Orthogonal direction of m2

w = sim.GetGaussianVector( 0, 1, N)         # Global readout

IctxA = sim.GetGaussianVector( 0, 1, N)     # Context input
IctxB = sim.GetGaussianVector( 0, 1, N)     # Context input

# Generate the matrix

m1 = y1 + rhom*IctxA + betam*w
n1 = IA + rhon*IctxA + betan*w

m2 = y2 + rhom*IctxB + betam*w
n2 = IB + rhon*IctxB + betan*w

structure[0,:] = m1
structure[1,:] = m2
structure[2,:] = n1
structure[3,:] = n2
structure[4,:] = IA
structure[5,:] = IB
structure[6,:] = y1
structure[7,:] = y2
structure[8,:] = w
structure[9,:] = IctxA
structure[10,:] = IctxB

M = ( np.outer( m1, n1 ) + np.outer( m2, n2 ) ) / N
R = g * sim.GetBulk ( N )
J = R + M

#### Phase 0: discard the transient

Z_start = sim.SimulateActivity ( t0, -m1 -m2 -n1 -n2, J, I = 0 )

#### CONTEXT A

print 'Context A'

#### Phase 1: no stimulus input

I = gammaON * IctxA + gammaOFF * IctxB 
Z = sim.SimulateActivity ( t1, Z_start[-1,:], J, I )

activation_A[0, 0:len(t1), :] = Z
activation_A[1, 0:len(t1), :] = Z

readout_A[0, 0:len(t1)] = np.dot(np.tanh(Z), w) / N
readout_A[1, 0:len(t1)] = np.dot(np.tanh(Z), w) / N

#### Phase 2: stimulus input

# A
I = gammaON * IctxA + gammaOFF * IctxB + IA
ZA = sim.SimulateActivity ( t2, Z[-1,:], J, I)
activation_A[0, len(t1):, :] = ZA
readout_A[0, len(t1):] = np.dot(np.tanh(ZA), w) / N

# B
I = gammaON * IctxA + gammaOFF * IctxB + IB
ZB = sim.SimulateActivity ( t2, Z[-1,:], J, I)
activation_A[1, len(t1):, :] = ZB
readout_A[1, len(t1):] = np.dot(np.tanh(ZB), w) / N

print readout_A[0,-1], readout_A[1,-1]


#### CONTEXT B

print 'Context B'

#### Phase 1: no stimulus input

I = gammaOFF * IctxA + gammaON * IctxB 
Z = sim.SimulateActivity ( t1, Z_start[-1,:], J, I )

activation_B[0, 0:len(t1), :] = Z
activation_B[1, 0:len(t1), :] = Z

readout_B[0, 0:len(t1)] = np.dot(np.tanh(Z), w) / N
readout_B[1, 0:len(t1)] = np.dot(np.tanh(Z), w) / N

#### Phase 2: stimulus input

# A
I = gammaOFF * IctxA + gammaON * IctxB + IA
ZA = sim.SimulateActivity ( t2, Z[-1,:], J, I)
activation_B[0, len(t1):, :] = ZA
readout_B[0, len(t1):] = np.dot(np.tanh(ZA), w) / N

# B
I = gammaOFF * IctxA + gammaON * IctxB + IB
ZB = sim.SimulateActivity ( t2, Z[-1,:], J, I)
activation_B[1, len(t1):, :] = ZB
readout_B[1, len(t1):] = np.dot(np.tanh(ZB), w) / N

print readout_B[0,-1], readout_B[1,-1]


# Store

fac.Store(ParVec, 'ParVec.p', path_data)

fac.Store(readout_A, 'readout_A.p', path_data)
fac.Store(readout_B, 'readout_B.p', path_data)

fac.Store(activation_A, 'activation_A.p', path_data)
fac.Store(activation_B, 'activation_B.p', path_data)

fac.Store(structure, 'structure.p', path_data)

fac.Store(t, 't.p', path_data)
fac.Store(t1, 't1.p', path_data)
fac.Store(t2, 't2.p', path_data)

#
