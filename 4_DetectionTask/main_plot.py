
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 4: sample rank-one network for the Go-Nogo detection (related to Fig. 4 B-C-D-E)
#### The code is composed of three main scripts, to be run in the order:
#### main_simulate.py  -  generates a network and the activity in response to four stimulus intensities,
####                      for Nic different noise realizations
#### main_plot.py  -  plots single neuron and population activity
#### main_regression.py  -  computes the regression axis for input and choice

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
activation = fac.Retrieve('activation.p', path_data)
readout = fac.Retrieve('readout.p', path_data)

ParVec = fac.Retrieve('ParVec.p', path_data)
c_values = fac.Retrieve('c_values.p', path_data)
sigma = fac.Retrieve('sigma.p', path_data)

t = fac.Retrieve('t.p', path_data)
t1 = fac.Retrieve('t1.p', path_data)
t2 = fac.Retrieve('t2.p', path_data)

Sii, Siw, rhom, rhon = fac.Retrieve('ParVec.p', path_data)

N = activation.shape[3]
I = structure[1,:]
m = structure[0,:] + ParVec[2]*structure[2,:]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot

fac.SetPlotParams()
dashes = [3,3]

color = np.flipud(np.array(['#2D3246', '#4170A6', '#EFAC07', '#D95D39']))
threshold = 0.54 # Value of the detection threshdold, computed through DMF theory

# Average input intensity

fac.SetPlotDim(2.1, 1.3)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for i, c in enumerate(c_values):

	c = np.concatenate ( [ 0. * np.ones(len(t1)), c * np.ones(len(t2)) ])
	plt.plot(t, c, color = color[i])

	line, = plt.plot(t, threshold * np.ones(len(t)), color = '#CC0000')
	line.set_dashes(dashes)

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Mean input')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 1.2)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('input_intensity.pdf')
plt.show()


# Sample of activation profiles, for a single trial Ic

fac.SetPlotDim(2.1, 4.)

Ic = 0

Nsample = 7

for i, c in enumerate(c_values):

	fg = plt.figure()
	ax0 = plt.axes(frameon=True)

	for k in range(Nsample):
		plt.plot(t, 6 + 7*k + activation[i, Ic,:,k], color = color[i] )

	plt.xlabel(r'time (norm.)')
	plt.ylabel(r'Activation $x_i$')

	plt.xlim(0, t[-1])
	plt.ylim(-0.1, 60)

	ax0.spines['top'].set_visible(False)
	ax0.spines['right'].set_visible(False)
	ax0.yaxis.set_ticks_position('left')
	ax0.xaxis.set_ticks_position('bottom')
	plt.locator_params(nbins=5)

	plt.savefig('activation_'+str(i)+'.pdf')
	plt.show()


# Sample of readout, for a single trial

fac.SetPlotDim(2.1, 1.85)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for i, c in enumerate(c_values):

	plt.plot(t, readout[i, Ic, :] - readout[i, Ic, 0], color = color[i] )

plt.xlabel(r'time (norm.)')
plt.ylabel(r'Readout $z$')

plt.xlim(0, t[-1])
plt.ylim(-0.1, 1.5)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('readout.pdf')
plt.show()


# Project on the m-I plane, for a single trial

fac.SetPlotDim(2.1, 2.1)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for i, c in enumerate(c_values):

	on_m = np.dot( activation[i, Ic, :], m ) / N
	on_I = np.dot( activation[i, Ic, :], I ) / N

	plt.plot(on_I, on_m, color = color[i])

plt.xlabel(r'$I$')
plt.ylabel(r'$m$')

plt.xlim(-3, 3)
plt.ylim(-10, 10)

plt.xticks([-3, 0, 3])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('plane.pdf')
plt.show()


# Project on the m-I plane, average over trials, Go responses

fac.SetPlotDim(2.1, 2.1)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for i, c in enumerate(c_values):

	on_m = np.dot( np.mean( activation[i, :, :][np.where(readout[i,:,-1]>0)], 0) , m ) / N
	on_I = np.dot( np.mean( activation[i, :, :][np.where(readout[i,:,-1]>0)], 0) , I ) / N

	plt.plot(on_I, on_m, color = color[i])

plt.xlabel(r'$I$')
plt.ylabel(r'$m$')

plt.xlim(-1, 3)
plt.ylim(-10, 10)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('plane_average_Go.pdf')
plt.show()


# Project on the m-I plane, average over trials, Nogo responses

fac.SetPlotDim(2.1, 2.1)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

for i, c in enumerate(c_values):

	on_m = np.dot( np.mean( activation[i, :, :][np.where(readout[i,:,-1]<0)], 0) , m ) / N
	on_I = np.dot( np.mean( activation[i, :, :][np.where(readout[i,:,-1]<0)], 0) , I ) / N

	plt.plot(on_I, on_m, color = color[i])

plt.xlabel(r'$I$')
plt.ylabel(r'$m$')

plt.xlim(-1, 3)
plt.ylim(-10, 10)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig('plane_average_Nogo.pdf')
plt.show()

#
