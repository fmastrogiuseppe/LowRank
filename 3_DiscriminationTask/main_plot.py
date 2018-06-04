
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 3: sample rank-one network for the Go-Nogo discrimination task (related to Fig. 3 B-C-E-F)
#### The code is composed of three main scripts, to be run in the order:
#### main.simulate.py  -  genates a network and the activity in response to Go and Nogo stimulus
#### main.plot.py  -  plots single neuron and population activity
#### main.PCA_connectivity.py  -  computes the first PC axis for Go and Nogo trials, and correlates
####                              them with the average pair connectivity


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
activation = fac.Retrieve('activation.p', path_data)
readout = fac.Retrieve('readout.p', path_data)

t = fac.Retrieve('t.p', path_data)
t1 = fac.Retrieve('t1.p', path_data)
t2 = fac.Retrieve('t2.p', path_data)
t3 = fac.Retrieve('t3.p', path_data)

Sii, Siw = fac.Retrieve('ParVec.p', path_data)

m = structure[0]
IA = structure[1]
IB = structure[2]
N = activation.shape[2]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot

fac.SetPlotParams()

color_Go = '#1C63A9'
color_Nogo = '#009999'


# Input intensity

fac.SetPlotDim(2.1, 1.3)

fg=plt.figure()
ax0 = plt.axes(frameon=True)

c = np.concatenate ( [ 0. * np.ones(len(t1)), 1 * np.ones(len(t2)), 0. * np.ones(len(t3)) ] )

plt.plot(t, c, color = '0.5')

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Input')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 2.)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('input_intensity.pdf')
plt.show()


# Sample of activation profiles - Go stimulus

fac.SetPlotDim(2.1, 4.)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

Nsample = 7

for i in range(Nsample):
	plt.plot(t, 3 + 5.5*i + activation[0,:,i], color = color_Go )

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Activation $x_i$')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 40)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.legend(loc = 1)

plt.savefig('activation_Go.pdf')
plt.show()


# Sample of activation profiles - Nogo stimulus

fac.SetPlotDim(2.1, 4.)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

Nsample = 7

for i in range(Nsample):
	plt.plot(t, 3 + 5.5*i + activation[1,:,i], color = color_Nogo )

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Activation $x_i$')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 40)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.legend(loc = 1)

plt.savefig('activation_Nogo.pdf')
plt.show()


# Readout

fac.SetPlotDim(2.1, 1.85)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot(t, readout[1,:], color = color_Nogo, label = '$I^A$')
plt.plot(t, readout[0,:], color = color_Go, label = '$I^B$')

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Readout $z$')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 2.)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.legend(loc = 2)

plt.savefig('readout.pdf')
plt.show()


# Project the Go trials on the m-IA plane

fac.SetPlotDim(2.1, 2.1)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

on_m = np.dot( activation, m ) / N
on_IA = np.dot( activation, IA ) / N
on_IB = np.dot( activation, IB ) / N

plt.plot(on_IA[0,:], on_m[0,:], color = color_Go)

plt.xlabel(r'$\delta I$')
plt.ylabel(r'$m$')

plt.xlim(-2, 4)
plt.ylim(-2, 6)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('plane_A.pdf')
plt.show()


# Project the Nogo trials on the m-IB plane

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot(on_IB[1,:], on_m[1,:], color = color_Nogo)

plt.xlabel(r'$\delta I$')
plt.ylabel(r'$m$')

plt.xlim(-2, 4)
plt.ylim(-2, 6)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('plane_B.pdf')
plt.show()

#

