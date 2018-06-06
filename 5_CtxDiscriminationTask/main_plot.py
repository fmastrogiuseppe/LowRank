
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
### Load data

path_data = 'Data/'

structure = fac.Retrieve('structure.p', path_data)
activation_A = fac.Retrieve('activation_A.p', path_data)
activation_B = fac.Retrieve('activation_B.p', path_data)
readout_A = fac.Retrieve('readout_A.p', path_data)
readout_B = fac.Retrieve('readout_B.p', path_data)

ParVec = fac.Retrieve('ParVec.p', path_data)

t = fac.Retrieve('t.p', path_data)
t1 = fac.Retrieve('t1.p', path_data)
t2 = fac.Retrieve('t2.p', path_data)

Sii, rhom, rhon, betam, betan = fac.Retrieve('ParVec.p', path_data)

N = activation_A.shape[2]
IA = structure[4,:]
IB = structure[5,:]
w = structure[8,:]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot
# We plot one example situation (presentation of stimulus A in both contexts)

fac.SetPlotParams()
dashes = [3,3]

color_stimuli = np.array(['#1C63A9', '#009999'])
color_ctx = np.array(['#7C003E', '#E56C80'])


# Input intensity

fac.SetPlotDim(2.1, 1.3)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cA = np.concatenate ( [ 0. * np.ones(len(t1)), 1 * np.ones(len(t2)) ])
plt.plot(t, cA, color = color_stimuli[0])

cB = np.concatenate ( [ 0. * np.ones(len(t1)), 0 * np.ones(len(t2)) ])
plt.plot(t, cB, color = color_stimuli[1])

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Stimulus')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 1.5)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('input_stimuli.pdf')
plt.show()


# Sample of activation profiles, for stimulus A in context A

fac.SetPlotDim(2.1, 4.)

Nsample = 7

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for k in range(Nsample):
	plt.plot(t, 6 + 7*k + activation_A[0, :, k], color = color_ctx[0] )

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Activation $x_i$')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 60)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('activation_A.pdf')
plt.show()


# Sample of activation profiles, for stimulus A in context B

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for k in range(Nsample):
	plt.plot(t, 6 + 7*k + activation_B[0, :, k], color = color_ctx[1] )

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Activation $x_i$')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 60)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('activation_B.pdf')
plt.show()


# Readouts, for stimulus A in both contexts

fac.SetPlotDim(2.1, 1.85)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot(t, readout_B[0, :] - readout_B[0, 0], color = color_ctx[1], label = 'Ctx $A$' )
plt.plot(t, readout_A[0, :] - readout_A[0, 0], color = color_ctx[0], label = 'Ctx $B$' )

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Readout $z$')

plt.legend(loc=2)

plt.xlim(0, t[-1])
plt.ylim(-0.1, 1.)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('readout.pdf')
plt.show()


# Project on the m-IA plane, in both contexts

fac.SetPlotDim(2.1, 2.1)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

on_w = np.dot( activation_A[0, :, :], w ) / N
on_IA = np.dot( activation_A[0, :, :], IA ) / N
plt.plot(on_IA, on_w, color = color_ctx[0], label = 'Ctx $A$')

on_w = np.dot( activation_B[0, :, :], w ) / N
on_IA = np.dot( activation_B[0, :, :], IA ) / N
plt.plot(on_IA, on_w, color = color_ctx[1], label = 'Ctx $B$')

plt.xlabel(r'$I^A$')
plt.ylabel(r'$w$')

plt.legend(loc=2)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('plane.pdf')
plt.show()


#
